import logging 
from tqdm import tqdm
import argparse 
from code.utils.utils import * 

from .data import get_all_grid_rotations, load_generated_data_info
from .data import ProgramData, PAD_IDX
from .networks import Synthesizer
from .sampling import beam_sample

from code.utils.parser.world import World  
from code.utils.parser.parser import Parser
from code.utils.parser.utils import get_num_blocks

import torch 
import numpy as np 
from torch.utils.data import DataLoader



def evaluate_metrics(loop_data, metrics, prog_idx, vocab):

    num_exact, num_gen, num_sem, num_min_sem, pred_top = metrics
    parser = Parser()

    for samples, target, spec_in, spec_out\
            ,test_in, test_out in loop_data:

            updated_corr, updated_sem, updated_gen, updated_min_sem = \
                False, False, False, False

            # remove pad 
            target = target.numpy().tolist()
            target_idxs = [idx for idx in target if idx != PAD_IDX]
            target_tokens = [vocab["idx2tkn"][idx] for idx in target]                           
            # for each ray in beam 
            for rank, dec in enumerate(samples):
                pred = dec[-1]
                pred = [idx for idx in pred if idx != PAD_IDX]
                    
                # save top prediction 
                if rank == 0 : 
                    pred_top.append(pred)

                if pred == target_idxs and not updated_corr:
                    num_exact[prog_idx,rank:] += 1
                    updated_corr = True


                pred = [vocab["idx2tkn"][idx] for idx in pred]
                pred =  ' '.join(pred)
 
                
                # if this dec is not parsable go to next 
                if not parser.parse(pred) : 
                    continue 

                sem = True 

                # check specification 
                for inp, out in zip(spec_in, spec_out):
                    
                    world = World()
                    world.read_from_state(inp)
                    world.run(pred)
                    
                    if world.crashed : 
                        sem = False 
                        break 

                    out_grid = world.get_state()
                    g_x, g_y = np.where(out_grid)[1][-1], np.where(out_grid)[2][-1]
                    a_x, a_y = np.where(out_grid)[1][0], np.where(out_grid)[2][0]
                    if  (a_x != g_x or g_y != a_y): 
                        sem = False
                        break  

                # no need to go forward if this is not 
                # at least semantically correct                   
                if not sem : 
                    continue 


                if sem and not updated_sem:
                    num_sem[prog_idx, rank:] += 1
                    updated_sem = True

                if sem and get_num_blocks( pred.split() ) <= get_num_blocks( target_tokens ) and not updated_min_sem:
                    num_min_sem[prog_idx, rank:] += 1 
                    updated_min_sem = True 
                gen = True
                for inp, out in zip(test_in, test_out):

                    world = World()
                    world.read_from_state(inp)
                    world.run(pred)

                    if world.crashed :
                        gen = False 
                        break 

                    out_grid = world.get_state()

                    g_x, g_y = np.where(out_grid)[1][-1], np.where(out_grid)[2][-1]
                    a_x, a_y = np.where(out_grid)[1][0], np.where(out_grid)[2][0]
                    if  (a_x != g_x or g_y != a_y): 
                        sem = False
                        break  


                if gen and not updated_gen: 
                    num_gen[prog_idx, rank:] += 1
                    updated_gen = True

                # if you found the first scoring rank for all metrics, break loop 
                if updated_corr and updated_sem and updated_gen and updated_min_sem : 
                    break  
            # next sample 
            prog_idx += 1

def evaluate_model(model, device, dataloader, vocab, top_k, beam_size, num_io_eval, rotate):


    total_num = len(dataloader.dataset)

    # exact match 
    num_exact = np.zeros((total_num, top_k))
    # semantically correct 
    num_sem = np.zeros((total_num, top_k))
    # generalization
    num_gen = np.zeros((total_num, top_k))
    # semantic and min length 
    num_min_sem = np.zeros((total_num, top_k))
    # save top prediction 
    pred_top = []

    metrics = [num_exact, num_gen, num_sem, num_min_sem, pred_top]



    prog_idx = 0 
    for batch in tqdm(dataloader):

        inp_grids,out_grids,in_tgt_seq,out_tgt_seq  = batch

        assert inp_grids.shape[1] > 1  

        in_spec = inp_grids[:,:num_io_eval,:].to(device)
        out_spec = out_grids[:,:num_io_eval,:].to(device)        
        
        if rotate :
            assert num_io_eval == 1
            in_spec = get_all_grid_rotations(in_spec)
            out_spec = get_all_grid_rotations(out_spec)


        in_test = inp_grids[:,-1:,:].to(device)
        out_test = out_grids[:,-1:,:].to(device)
        
        if rotate : 
            in_test = get_all_grid_rotations(in_test)
            out_test = get_all_grid_rotations(out_test)

        max_len = out_tgt_seq.size(1) + 10
        
        # decoded :: BS x topk x ([seqs],probs) 
        with torch.no_grad():
            decoded = beam_sample(model, max_len,in_spec,out_spec,
                beam_size, top_k)


        in_spec = in_spec.cpu().numpy().astype(int) 
        out_spec = out_spec.cpu().numpy().astype(int)
        in_test = in_test.cpu().numpy().astype(int)
        out_test = out_test.cpu().numpy().astype(int)



        loop_data = zip(decoded,out_tgt_seq,in_spec,out_spec,in_test,out_test)
        evaluate_metrics(loop_data, metrics, prog_idx, vocab)

        prog_idx += len(batch[0])

    num_exact = 100*np.mean(num_exact,axis =0) 
    num_sem = 100*np.mean(num_sem,axis =0)
    num_gen = 100*np.mean(num_gen,axis =0) 
    num_min_sem = 100*np.mean(num_min_sem,axis =0)

    logging.info(f'Exatch match {num_exact}')
    logging.info(f'Semantically correct {num_sem}')
    logging.info(f'Generalization {num_gen}')
    logging.info(f'Semantically & min length {num_min_sem}')
    
    return num_exact, num_sem, num_gen, num_min_sem 

def eval_main():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data', required = True)
    arg_parser.add_argument('--batch_size', default = 32, type = int)
    arg_parser.add_argument('--device', default = 'cuda')
    arg_parser.add_argument('--run_id', required = True)
    arg_parser.add_argument('--task', required = True)
    arg_parser.add_argument('--top_k', type = int, default = 1)
    arg_parser.add_argument('--beam_size', type = int, default = 64)
    arg_parser.add_argument('--num_io_eval', type = int, default = 1)
    arg_parser.add_argument("--rotate",type = bool , default = True, help = 'use all 4 rotations of a grid as input')
    arg_parser.add_argument("--override",action = 'store_const', default = False, const = True, help ='ignore cached data, read input again')

    args = arg_parser.parse_args()

    args.data = NEURSS_DIR + f'/{args.task}/training_data/{args.data}/'
    args.test_data =  args.data + 'test.data'
    
    
    data_info = load_generated_data_info(args.data + '/info.json')
    test_dataset = ProgramData(
        args.test_data, data_info, rotate = False, 
        override = args.override
        )
    test_dataloader = DataLoader(test_dataset, batch_size= args.batch_size,
                            shuffle=False)

    load_path = NEURSS_DIR + f'/{args.task}/runs/{args.run_id}/model_best.pt'

    device = args.device
    checkpoint = torch.load(load_path, map_location=device)


    model_args = checkpoint['model_args']
    model = Synthesizer(**model_args)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().to(device)

    num_exact, num_sem, num_gen, num_min_sem  = evaluate_model(model, device, test_dataloader, data_info["vocab"], 
        args.top_k, args.beam_size, args.num_io_eval, args.rotate)

    print(f'Exatch match {num_exact}')
    print(f'Semantically correct {num_sem}')
    print(f'Generalization {num_gen}')
    print(f'Semantically & min length {num_min_sem}')



if __name__ == "__main__" : 
    eval_main()