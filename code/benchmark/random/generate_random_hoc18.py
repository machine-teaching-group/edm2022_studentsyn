from copy import deepcopy 
import numpy as np 
import json 
import pandas as pd 
import random 
from code.utils.utils import * 
from .utils_random import calculate_edit_distance, action_tokens, conditions, remove_if, remove_while
from code.utils.convert import ast_code_to_token_list, token_list_to_benchhmark_ast
from .generate_random_hoc4 import edit_code_hoc4
import argparse
import os 

def util_edit_hoc18(sol_code_edited, num_edits, 

    p_add = {"move" : 1/3, "turn_left" : 1/3, "turn_right" : 1/3}, 
    p_select = {"add" : 1/3, "remove" : 1/3, "transform" : 1/3}):

    retokenize = lambda code : ' '.join(code).split()

    sol_code_edited = np.array(sol_code_edited)
    # select idxs that we can transform for edit-distance
    candidate_idxs = np.where(
        (sol_code_edited == 'turn_left') | (sol_code_edited == 'turn_right') |(sol_code_edited == 'move') | 
        (sol_code_edited == 'bool_path_left') | (sol_code_edited == 'bool_path_ahead') | (sol_code_edited == 'bool_path_right') |
        (sol_code_edited == 'm(') | (sol_code_edited == 'w(')  |   (sol_code_edited == 'e)')   
        )[0]

    sol_code_edited = sol_code_edited.tolist()
    # select actual from candidate
    idxs_change = np.random.choice(candidate_idxs, size = num_edits, replace = True)
    # can't transform after delete etc. 
    previous_actions = {idx : set() for idx in candidate_idxs} 
    changed_bool = False 

    for idx in idxs_change : 
        # handle special case
        if 'bool' in sol_code_edited[idx]:
            if changed_bool == True : 
                # sample a new one 
                idxs_cand_new = deepcopy(candidate_idxs.tolist())
                idxs_cand_new.remove(idx)
                idx = np.random.choice(idxs_cand_new)
       
            else : 
                # change the condition and move on 
                tokens_sample = deepcopy(conditions)
                tokens_sample.remove(sol_code_edited[idx])
                sol_code_edited[idx] = np.random.choice(tokens_sample)    
                changed_bool = True 
                continue   
        
        choices = deepcopy(['remove', 'add', 'transform'])
        # if removing causes an error don't remove 
        try :
            sol_code_removed = deepcopy(sol_code_edited)
            sol_code_removed[idx] = ''
            
            token_list_to_benchhmark_ast(retokenize(sol_code_removed))
        except : 
            #print('caught')
            choices.remove('remove')

        # can't transform and remove a token    
        if 'transform' in previous_actions[idx] : choices = ['add']
        if 'remove' in previous_actions[idx] : choices = ['add']            

        # can only add to these candidates    
        for token in sol_code_edited[idx].split():
            if token in ['m(', 'w(', 'e)']:
                choices = ['add']
                break 

        # choose your edit 
        probs = [p_select[choice] for choice in choices]
        probs = np.array(probs)  / sum(probs)
        edit_action = np.random.choice(choices, p = probs)

        previous_actions[idx].add(edit_action)   

        actions_sample = deepcopy(action_tokens)

        if edit_action == 'remove' :
            sol_code_edited[idx] = ''
            
        elif edit_action == 'add' : 
            p_action = [p_add[t] for t in actions_sample]                
            action_extra = np.random.choice(actions_sample, p = p_action).tolist()
            sol_code_edited[idx] += ' ' + action_extra     
   
        elif edit_action == 'transform':
   
            tokens = sol_code_edited[idx].split()
            token_original = tokens[0]
   
            if token_original in actions_sample:
                actions_sample.remove(token_original)
   
            tokens[0] = np.random.choice(actions_sample)    
            sol_code_edited[idx] = ' '.join(tokens)

    sol_code_edited = ' '.join(sol_code_edited).split()
    return sol_code_edited


def edit_code_hoc18(sol_code_tokens, num_edits, p_select , p_add) : 
    
    edit_dist = 0 
    while edit_dist != num_edits :
        sol_code_edited = deepcopy(sol_code_tokens)
        sol_code_edited = util_edit_hoc18(sol_code_edited, num_edits, p_select = p_select , p_add = p_add)
        edit_dist = calculate_edit_distance(sol_code_edited, sol_code_tokens)
        #print(sol_code_edited, edit_dist, num_edits)

    return sol_code_edited


def remove_tokens(sol_code_tokens, tokens) : 
    sol_code_str = ' '.join(sol_code_tokens)
    for remove_token in tokens :
        sol_code_str = sol_code_str.replace(remove_token, '')
    sol_code_tokens = sol_code_str.split()
    return sol_code_tokens



EDITS_BASIC = 5
EDITS_ONLY_IF = 2
EDITS_ONLY_WHILE = 2
EDITS_SOL = 2

def generate_random_codes_hoc18():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--random_per_type', type = int, default = 10)

    args = parser.parse_args()
    num_random_per_dist = args.random_per_type

    random.seed(args.seed) 
    np.random.seed(args.seed)

    source_task = 'hoc18'
    save_dir = BENCHMARK_DIR + f'/{source_task}/rand/'

    os.makedirs(save_dir, exist_ok=True)


    target_tasks = [f"{source_task}{indicator}" for indicator in ['','a','b','c']]

    random_codes_idx = 1
    random_code_df_rows = []
    random_codes = []

    for target_task in target_tasks : 
        print(target_task)
        # load basic action solution
        suffix = '_code_solution_basic_actions.json'
        sol_path_basic_actions = BENCHMARK_DIR + '/' + source_task + f'/solution_basic_actions/{target_task}{suffix}'
        with open(sol_path_basic_actions,'r') as f :
            sol_code_basic_actions = ast_code_to_token_list(json.load(f))

        suffix = '_code_solution.json'
        sol_path = BENCHMARK_DIR + '/' + source_task + f'/solution/{target_task}{suffix}'
        with open(sol_path,'r') as f :
            code_ast = json.load(f)
            sol_code_tokens = ast_code_to_token_list(code_ast)


        print('generating 2 edit-dist codes')
        for _ in range(num_random_per_dist):
            
            p_add = {"move" : 1/2, "turn_left" : 1/4, "turn_right" : 1/4}            
            p_select = {"add" : 4/10, "remove" : 2/10, "transform" : 4/10}

            sol_code = deepcopy(sol_code_tokens)

            sol_code_edited = edit_code_hoc18(sol_code, EDITS_SOL, p_add = p_add, p_select = p_select)

            random_code_df_rows.append([random_codes_idx, target_task, f'edit_{EDITS_SOL}'])
            random_codes.append(sol_code_edited)
            random_codes_idx+=1    

        print('generating only_if codes')
        for _ in range(num_random_per_dist):
            
            p_add = {"move" : 1/2, "turn_left" : 1/4, "turn_right" : 1/4}            
            p_select = {"add" : 4/10, "remove" : 2/10, "transform" : 4/10}

            sol_code = deepcopy(sol_code_tokens)
            sol_code_removed = remove_tokens(sol_code_tokens, remove_while)
            sol_code_edited = edit_code_hoc18(sol_code_removed, EDITS_ONLY_IF, p_add = p_add, p_select = p_select)            
            
            random_code_df_rows.append([random_codes_idx, target_task, 'only_if'])
            random_codes.append(sol_code_edited)
            random_codes_idx+=1    

        print('generating only_while codes')
        for _ in range(num_random_per_dist):
            
            p_add = {"move" : 1/2, "turn_left" : 1/4, "turn_right" : 1/4}            
            p_select = {"add" : 4/10, "remove" : 2/10, "transform" : 4/10}

            sol_code = deepcopy(sol_code_tokens)
            sol_code_removed = remove_tokens(sol_code, remove_if)
            sol_code_edited = edit_code_hoc18(sol_code_removed, EDITS_ONLY_WHILE, p_add = p_add, p_select = p_select)
        
            random_code_df_rows.append([random_codes_idx, target_task, 'only_while'])
            random_codes.append(sol_code_edited)
            random_codes_idx+=1    

        print('generating basic_action codes')
        for _ in range(num_random_per_dist):

            p_add = {"move" : 1/2, "turn_left" : 1/4, "turn_right" : 1/4}            
            p_select = {"add" : 2/10, "remove" : 4/10, "transform" : 4/10}
            sol_code = deepcopy(sol_code_basic_actions)
            sol_code_edited = edit_code_hoc4(sol_code, num_edits = EDITS_BASIC, 
                replace = False, exact = True, p_select = p_select , p_add = p_add)    

            random_code_df_rows.append([random_codes_idx, target_task, 'basic_actions'])
            random_codes.append(sol_code_edited)
            random_codes_idx+=1    
        


    random_code_df = pd.DataFrame(random_code_df_rows, columns = ["rand_id", "task", "notes"])
    random_code_df.to_csv(BENCHMARK_DIR + f'/{source_task}/' + 'rand_info.tsv', sep = '\t', index=False)

    for code, row in zip(random_codes, random_code_df_rows) :     

        rand_idx, rand_task, _ = row
        path = save_dir + f'{rand_task}_rand{rand_idx}.json'
        code_ast = token_list_to_benchhmark_ast(code)


        with open(path, 'w+') as f :
            json.dump(code_ast, f, indent = 2)



if __name__ == '__main__' : 
    '''
        to generate new random codes, first delete the folder in data/benchmark/hoc18/rand/
    '''
    generate_random_codes_hoc18()