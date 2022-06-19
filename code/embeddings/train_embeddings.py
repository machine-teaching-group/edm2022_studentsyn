import numpy as np
from tqdm import tqdm 
import pickle
import os
import argparse 
import json

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from .utils import dataset_access
from .utils import ast_utils
from .data import StudentData, SyntaxData, get_embed_tensor, get_vocab_size, HIDDEN_DIM
from code.utils.utils import EMB_DIR, set_logger

import logging


class DistanceRNN(nn.Module):

    def __init__(self,vocab_size , hidden_dim):

        super(DistanceRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size,hidden_dim, padding_idx = 0)       
        self.lstm = nn.LSTM(input_size = hidden_dim, hidden_size = hidden_dim, 
            num_layers =2 , bidirectional = True, batch_first = True)

        hidden_dim = hidden_dim * 2 

        # first encoder 
        self.nonlinear_in = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh()
        )
        self.scale_in = nn.Linear(hidden_dim,hidden_dim)

        # second encoder 
        self. nonlinear_out = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh()
        )
        self.scale_out = nn.Linear(hidden_dim,hidden_dim)

        # scaling module, for comparing between enc1, enc2 for distance
        self. nonlinear_d = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh()
        )
        self.scale_d = nn.Linear(hidden_dim,hidden_dim)

    def d(self,x,y):
        # x : outs, y : ins
        # compare two hidden states 
        # scale one of them 
        h_x = self.scale_d(self.nonlinear_d(x))
        return torch.norm(h_x - y, dim = -1)

    def forward(self,x, y):

        len_x = torch.bincount(torch.where(x != 0)[0]).cpu()
        len_y = torch.bincount(torch.where(y != 0)[0]).cpu()
        s_x = self.embedding(x)
        s_y = self.embedding(y)

        ins1, outs1 = self.inout(s_x, len_x)
        ins2, outs2 = self.inout(s_y, len_y)

        distances12 = self.d(outs1, ins2)
        distances21 = self.d(outs2, ins1)

        return distances12, distances21


    def inout(self, x, len_x):

        # pack sequence 
        x_packed = pack_padded_sequence(x, len_x,batch_first=True, enforce_sorted = False)
        # pass through rnn 
        state, _  = self.lstm(x_packed)        

        # get output from packed_sequence
        # index at correct place using lens
        output, lengths = pad_packed_sequence(state, batch_first=True)

        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(
            len(lengths), output.size(2)
        )
        idx = idx.unsqueeze(1).to(output.device)
        last_output = output.gather(1, idx).squeeze(1)


        in_emb = self.scale_in(self.nonlinear_in(last_output))
        out_emb = self.scale_out(self.nonlinear_out(last_output))

        return in_emb, out_emb



def loss_fn(model, batch, device):

    loss = torch.nn.MSELoss()
    x, y, ds12, ds21 = batch 
    x = x.to(device)
    y = y.to(device)
    ds12 = ds12.to(device)
    ds21 = ds21.to(device)

    distances12, distances21 = model(x, y)
    return 0.5 * loss(ds12.float(), distances12) + 0.5 * loss(ds21.float(), distances21)


def train( epochs, model, optimizer, train_loader,save_dir, val_loader = None, dataset_id='hoc4'):

    
    val_best = float('Inf')  
    device = next(model.parameters()).device
    for i in range(0, epochs):

        # train model 
        with tqdm(train_loader,unit = 'batch') as tepoch :  
            for batch in tepoch : 
                optimizer.zero_grad()                
                loss = loss_fn(model,batch, device)
                loss.backward()

                optimizer.step() 
                tepoch.set_postfix(loss=loss.item())

        # try to validate if validation is present 
        if val_loader is not None:
        
            with torch.no_grad():
                model.eval()
                losses = []
                for batch in val_loader : 
                    val_loss = loss_fn(model, batch, device) 
                    losses.append(val_loss.item())
                val_loss = np.mean(losses)
                logging.info(f'Epoch {i+1}, val_loss : {val_loss}')

                if val_loss < val_best :
                    val_best = val_loss 
                    state = {
                            "state_dict": model.state_dict(),
                            "optimizer_dict" : optimizer.state_dict(),
                            "epoch" : i
                            } 
                    save_path = save_dir + f'{dataset_id}_best.pt' 
                    torch.save(state, save_path)
                    logging.info('saving best model')
                model.train()                     





def train_rnn(rnn, args):

    # validate date 
    val_data = SyntaxData(args.dataset_id, 'validate')
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size,
                        shuffle=True)
    # train data 
    train_data = SyntaxData(args.dataset_id, 'train')
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,
                        shuffle=True)

    optimizer = torch.optim.Adam(rnn.parameters(),lr = args.lr)
 
    train(
        epochs=args.epochs,
        model=rnn,
        optimizer= optimizer, 
        train_loader = train_dataloader,
        save_dir = args.save_dir,
        val_loader = val_dataloader,
        dataset_id = args.dataset_id
    )

    return rnn, optimizer 

def embed_tokens_util(tokens, embed_tensor):
    tokens = [torch.tensor(t) for t in tokens]
    x = pad_sequence(tokens, batch_first = True)
    A = embed_tensor
    orig1 = x.shape
    x = (x.view(-1,1) == A).int().argmax(dim=1).reshape(orig1) 
    return x

def embed_tokens(tokens, model, embed_tensor):
    '''
        tokens :: list of ast_token idxs 
        like in ast_utils

    '''
    device = next(model.parameters()).device
    x = embed_tokens_util(tokens , embed_tensor)
    len_x = torch.bincount(torch.where(x != 0)[0]).cpu()
    x = x.to(device)
    x = model.embedding(x)
    ins, outs = model.inout(x,len_x)
    emb = torch.cat((ins,outs), dim = -1)
    return emb 

def create_embeddings(task, model, device, model_dir):

    
    name = f'{task}_data_embedded.p'
    embed_tensor = get_embed_tensor[task]

    data = StudentData(task)
    dataloader = DataLoader(data, batch_size=512, collate_fn=lambda x: x)    
    nodes = []
    embeddings = []
    ast_idxs = []
    model.to(device)
    model.eval()

    asts_all = []

    for batch in dataloader : 
        idxs  = [b[0] for b in batch] 
        asts  = [b[1] for b in batch] 
        nodes_new = []
        asts_new = []
        for code_ast in asts : 
            try : 
                nodes_new.append(ast_utils.ASTNode.from_json(code_ast))
            except : 
                print('strange student code, ignoring')
                continue 
            asts_new.append(code_ast) 

        asts_all += asts_new
        tokens = [node.tokenize() for node in nodes_new]
        embs =  embed_tokens(tokens, model, embed_tensor) 
        embeddings.append(embs)

    embeddings = torch.cat(embeddings).cpu().numpy()

    dataset = {
        "embeddings" : embeddings, 
        "asts" : asts_all
    }

    print(f'created embeddings for {task}, saving..')
    with open(dataset_access.GENERATED_DIR + f'/{task}/{model_dir}/'  + name,'wb') as handle : 
        pickle.dump(dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_embeddings(task, model_dir):
    name = f'{task}_data_embedded.p'
    with open(dataset_access.GENERATED_DIR + f'/{model_dir}/' + name,'rb') as handle : 
        dataset = pickle.load(handle)

    return dataset["features"], dataset["ast_2_node_id"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task',default = 'hoc4')
    parser.add_argument('--dim',default = HIDDEN_DIM, type = int)
    parser.add_argument('--batch_size',default = int(2**7), type = int)
    parser.add_argument('--device',default = 'cuda')
    parser.add_argument('--epochs',type = int, default = '100')
    parser.add_argument('--lr',type = float, default = 1e-4)
    parser.add_argument('--cmd',type = str, required = True, choices = ['train', 'create_emb', 'load_emb'])
    parser.add_argument('--model_dir', default = 'model_1', type = str)

    args = parser.parse_args()

    save_dir =  EMB_DIR + f'/{args.task}/' + args.model_dir + '/'
 
    args.save_dir = save_dir
    args.dataset_id = args.task
    vocab_size = get_vocab_size[args.task]
    model = DistanceRNN(vocab_size, args.dim ).to(args.device)
    if args.cmd == 'train':
        os.makedirs(args.save_dir, exist_ok = True)
        set_logger(args.save_dir + f'log_{args.task}.txt')
        with open(args.save_dir + 'args.json', 'w') as f : 
            json.dump(args.__dict__, f, indent= 2)

        train_rnn(model, args)

    elif args.cmd == 'create_emb' :

        path_to_model = args.save_dir + f'{args.task}_best.pt' 
        checkpoint = torch.load(path_to_model)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        with torch.no_grad() :
            create_embeddings(args.task, model , args.device, args.model_dir)

    elif args.cmd == 'load_emb' :   
        feats, dico = load_embeddings(args.task, args.model_dir)
