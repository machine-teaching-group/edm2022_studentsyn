import numpy as np 
import torch 
from code.HOC.PCFG.utils_train import transform_codes, generate_target_task_grammar
from code.HOC.utils import *


def NPS_token_idxs_to_string(codes, vocab):
    map_dictionary = vocab["idx2tkn"]
    assert torch.is_tensor(codes)
    uniq_codes ,inverse = np.unique(codes,return_inverse = True)
    ret = np.array([map_dictionary[x] for x in uniq_codes])[inverse].reshape(codes.shape)
    return ret 

def transform_codes(codes, vocab, pcfg_terminal_2_idx):
        clear_tokens = set(['<s>','m(','m)','<pad>','DEF','run'])
        codes_str = NPS_token_idxs_to_string(codes, vocab)
        code_tokens_all = []
        code_indexes_all = []
        for seq in codes_str:
            code_tokens_filtered =  [t for t in seq if t not in clear_tokens]        
            code_idxs_pcfg = [pcfg_terminal_2_idx[token] for token in code_tokens_filtered]
            code_tokens_all.append(code_tokens_filtered)
            code_indexes_all.append(code_idxs_pcfg)
        return code_indexes_all, code_tokens_all


def score_PCFG(G_source, data_B, vocab, source_task, args, label, code_A):
    task_paths, task_B_name , codes_B = data_B
    path = task_paths[task_B_name]['task_path']
    target_task = path.split('/')[-1].split('.')[0].split('_')[0]

    probs_per_misc = []
    for misc_id in range(6) : 
        G_new = generate_target_task_grammar(source_task, target_task, misc_id, vocab)
        tokens_pcfg, tokens_nps = transform_codes(codes_B, args.vocab, G_new.T2idx)
        probs = []
        lens = []

        for code in tokens_pcfg : 
            p,l = G_new.score(code, 'viterbi')
            probs.append(p)
            lens.append(l)

        lens = np.array(lens)
        lens[np.where(lens == 0)] = 1
        
        with np.errstate(divide='ignore'):
            probs_norm = np.log(np.array(probs))   / lens
        probs_norm = np.exp(probs_norm)
        probs_per_misc.append(probs_norm.tolist())

    probs_per_misc = np.array(probs_per_misc)
        
    pred = np.argmax(probs_per_misc[G_source.misconception])    
    return pred , probs_norm 
