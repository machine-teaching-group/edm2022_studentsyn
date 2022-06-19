import torch 
from torch import optim 
from torch import nn 
from torch.utils.data import DataLoader
from code.neurSS.synthesizer.networks import Synthesizer

def train_model_for_student(epochs, dataset, model ,device, lr, batch_size, data_info):
    dataloader = DataLoader(dataset, batch_size= batch_size,
                            shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)        
    train_loop(epochs, model ,device, data_info, dataloader , optimizer)

def train_loop(epochs,  model ,device, data_info, dataloader , optimizer):

    model.train()
    model.to(device)

    weight_mask = torch.ones(data_info["vocab_size"], device = device)
    weight_mask[data_info["pad_idx"]] = 0

    criterion = nn.CrossEntropyLoss(weight = weight_mask)
    for epoch in range(epochs):

        epoch_loss = []
        for batch in dataloader :   
            optimizer.zero_grad()

            inp_grids, out_grids, in_tgt_seq, out_tgt_seq  = batch 

            inp_grids = inp_grids.to(device)
            out_grids = out_grids.to(device) 
            out_tgt_seq = out_tgt_seq.to(device)
            in_tgt_seq = in_tgt_seq.to(device)  
        
            output, _, _, _ = model(inp_grids, out_grids, in_tgt_seq)
            gold = out_tgt_seq.flatten()
            pred = output.reshape(-1,output.shape[-1])
            loss = criterion(pred, gold)

            loss.backward() 
            optimizer.step()

            loss_item = loss.item() 
            epoch_loss.append(loss_item)


def get_model_for_training(load_path, model_freeze):
    return FineTuneSynth(load_path, model_freeze)

class FineTuneSynth(nn.Module):
    '''
        Use existing Synthesizer model with controlled finetuning 
    '''
    def __init__(self, load_path, model_freeze):
        super(FineTuneSynth, self).__init__()
        synthesizer, synth_args = prepare_synth_model(
            load_path,
            model_freeze
        )
        super(FineTuneSynth,self).add_module('synthesizer',synthesizer)
        self.check_syntax_mode = self.synthesizer.check_syntax_mode
        self.synth_args = synth_args

    def forward(self, inp, out, seq, lstm_states = None, grammar_state = None):
        return self.synthesizer(inp, out, seq, lstm_states, grammar_state)


def prepare_synth_model(load_path, model_freeze):

    load_path = load_path + f'model_best.pt'
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
    model_args = checkpoint['model_args']
    model = Synthesizer(**model_args)
    model.load_state_dict(checkpoint["state_dict"])

    if model_freeze == 'encoder' :  
        for param in model.encoder.parameters():
            param.requires_grad = False     

    elif model_freeze == 'encoder_no_emb' :
        for param in model.encoder.parameters():
            param.requires_grad = False 

        for param in model.encoder.embedding.parameters(): 
            param.requires_grad = True 

    elif model_freeze == 'none': 
        pass 

    elif model_freeze == 'no_emb' : 
        for param in model.parameters():
            param.requires_grad = False 
        for param in model.encoder.embedding.parameters(): 
            param.requires_grad = True 

    elif model_freeze == 'all' : 
        for param in model.parameters():
            param.requires_grad = False 
    else : 
        print('error, specify freezing')
        exit(1)


    return model, model_args

