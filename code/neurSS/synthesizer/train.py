from .networks import Synthesizer
from .evaluation import evaluate_model
from .arguments import parse_arguments
from .data import ProgramData

from code.utils.utils import * 

import logging 
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader
from torch import nn 
from torch import optim 

def train():

    args, model_args, data_info = parse_arguments()

    device = args.device 

    model = Synthesizer(**model_args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)        
    model.train()
    model.to(device)
    

    train_dataset = ProgramData(
        args.train_data, data_info, 
        args.num_io_train, args.rotate, override = args.override
        )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers = 10)

    val_dataset = ProgramData(
        args.val_data, data_info, 
        args.num_io_train, args.rotate, override = args.override
        )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                        shuffle=False)


    model_eval = Synthesizer(**model_args).eval().to(device)

    weight_mask = torch.ones(data_info["vocab_size"], device = device)
    weight_mask[data_info["pad_idx"]] = 0

    criterion = nn.CrossEntropyLoss(weight = weight_mask)

    best_validation = -1

    for epoch in range(args.epochs):

        epoch_loss = []

        with tqdm(train_dataloader,unit = 'batch') as tepoch :  

            for batch in tepoch :   
                optimizer.zero_grad()

                inp_grids, out_grids, in_tgt_seq, out_tgt_seq  = batch 

                inp_grids = inp_grids.to(device)
                out_grids = out_grids.to(device) 
                out_tgt_seq = out_tgt_seq.to(device)
                in_tgt_seq = in_tgt_seq.to(device)  

                if args.syntax == 'learned':
                    output, _, _, syntax_mask = model(inp_grids, out_grids, in_tgt_seq)

                    gold = out_tgt_seq.flatten()
                    pred = output.reshape(-1, output.shape[-1] )
                    ce_loss = criterion(pred,gold)
                    syntax_loss = -syntax_mask.gather(2, out_tgt_seq.unsqueeze(2)).sum()
                    loss = ce_loss + args.beta_syntax*syntax_loss
                else : 
                    output, _, _, _ = model(inp_grids, out_grids, in_tgt_seq)
                    gold = out_tgt_seq.flatten()
                    pred = output.reshape(-1,output.shape[-1])
                    loss = criterion(pred, gold)
 
                loss.backward() 
                optimizer.step()

                loss_item = loss.item() 
                tepoch.set_postfix(loss=loss_item)
                epoch_loss.append(loss_item)

        logging.info(f'mean epoch {epoch} loss: {loss:.2f}')

        if (epoch+1) % args.val_freq == 0:
            state = {
                "state_dict": model.state_dict(),
                "epoch" : epoch, 
                "model_args" :  model_args,
                "optimizer_dict" : optimizer.state_dict()
            } 

            model_eval.load_state_dict(state["state_dict"])

            num_exact, num_sem, num_gen, num_min_sem = evaluate_model(
                model_eval, device, 
                val_dataloader, data_info["vocab"], args.top_k, 
                args.beam_size, args.num_io_eval, args.rotate)
            logging.info(f'validating with {args.val_metric}')
            val_acc = {"sem" : num_sem[0], "exact" : num_exact[0],
                       "gen" : num_gen[0], "min_sem" : num_min_sem[0]}[args.val_metric]

            if val_acc > best_validation : 
                torch.save(state, args.save_path + f'/model_best.pt')
                best_validation = val_acc

            logging.info(f'\t: {epoch} Current        : {val_acc}')
            logging.info(f'\t: {epoch} ValidationBest : {best_validation}')

    test_dataset = ProgramData(
        args.test_data, data_info, 
        num_io = args.num_io_train, rotate = args.rotate, 
        override = args.override
        )
    test_dataloader = DataLoader(test_dataset, batch_size= args.batch_size,
                            shuffle=False)


    checkpoint = torch.load(args.save_path + '/model_best.pt', map_location=args.device)
    model_eval.load_state_dict(checkpoint["state_dict"])
    num_exact, num_sem, num_gen, num_min_sem = evaluate_model(
        model_eval, device, 
        test_dataloader, data_info["vocab"], args.top_k, 
        args.beam_size, args.num_io_eval, args.rotate)

    test_acc = {"sem" : num_sem[0], "exact" : num_exact[0],
                "gen" : num_gen[0], "min_sem" : num_min_sem[0]}[args.val_metric]
    logging.info(f'Test accuracy : {test_acc}')



if __name__ == '__main__' : 
    train()

