from copy import deepcopy 
import numpy as np 
import json 

import pandas as pd 
import argparse

from code.utils.utils import * 
from .utils_random import calculate_edit_distance, action_tokens
from code.utils.convert import ast_code_to_token_list, token_list_to_benchhmark_ast

import random 


def edit_code_hoc4(sol_code, num_edits, 
        p_select = {"add" : 1/3, "remove" : 1/3, "transform" : 1/3},
        p_add = {"move" : 1/3, "turn_left" : 1/3, "turn_right" : 1/3},
        p_trans = {"move" : 1/3, "turn_left" : 1/3, "turn_right" : 1/3},    
        replace = False, exact = True):

    edit_dist = 0 
    sol_code = sol_code[3:-1]  

    while edit_dist != num_edits : 

        solution_positions = range(len(sol_code))

        idxs_change = np.random.choice(solution_positions, 
            size = num_edits, 
            replace = replace)

        previous_actions = {sol_position : set() for sol_position in solution_positions} 
        sol_code_edited = deepcopy(sol_code)        

        retokenize = lambda code : ' '.join(code).split()
        normalize = lambda array :  np.array(array) / sum(array)
        for idx in idxs_change : 

            choices = ['remove', 'add', 'transform']
            probs = [
                        p_select["remove"], 
                        p_select["add"], 
                        p_select["transform"]
                    ]

      
            if len(retokenize(sol_code_edited)) == 1 : 

                choices = ["add", "transform"]
                probs = [p_select["add"], p_select["transform"]]
                probs = np.array(probs) / sum(probs)

            if 'transform' in previous_actions[idx] or 'remove' in previous_actions[idx]: 

                choices = ['add']
                probs = [1]

            edit_action = np.random.choice(
                    choices,  
                    p = probs)  
   
            previous_actions[idx].add(edit_action) 

            if edit_action == 'remove':

                sol_code_edited[idx] = ''
            
            elif edit_action == 'add': 

                prob_action = [p_add[action] for action in action_tokens]                

                action_extra = np.random.choice(action_tokens, 
                    p = prob_action).tolist()
         
                sol_code_edited[idx] += ' ' + action_extra  
    
            elif edit_action == 'transform':

                position_tokens = sol_code_edited[idx].split()
                original_token = position_tokens[0]  

                prob_trans_normalized = [p_trans[action] for action in action_tokens
                    if action != original_token]

                actions_trans = [action for action in action_tokens 
                    if action != original_token]

                prob_trans_normalized = normalize(prob_trans_normalized)

                position_tokens[0] = np.random.choice(actions_trans, 
                    p = prob_trans_normalized)
                
                sol_code_edited[idx] = ' '.join(position_tokens)


        sol_code_edited = retokenize(sol_code_edited)
        edit_dist = calculate_edit_distance(sol_code_edited, sol_code)
        if not exact : break  

    sol_code_edited = ' '.join(sol_code_edited).split()
    sol_code_edited = ['DEF', 'run', 'm('] + sol_code_edited + ['m)']


    return sol_code_edited


def generate_random_codes_hoc4():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--random_per_type', type = int, default = 10)

    args = parser.parse_args()
    random.seed(args.seed) 
    np.random.seed(args.seed)

    save_dir = BENCHMARK_DIR + f'/hoc4/rand/'
    target_tasks = [f"hoc4{indicator}" for indicator in ['','a','b','c']]

    random_code_df_rows = []
    random_codes = []
    random_codes_idx = 1

    num_random_per_dist = 10

    p_add = {"move" : 3/5, "turn_left" : 1/5, "turn_right" : 1/5}            
    
    for target_task in target_tasks  : 
        print(target_task)
        suffix = '_code_solution.json'
        sol_path = BENCHMARK_DIR + '/' + 'hoc4' + f'/solution/{target_task}{suffix}'

        with open(sol_path,'r') as f :
            solution_ast = json.load(f)
            solution_tokens = ast_code_to_token_list(solution_ast)


        for edit_dist in [2, 5, 10] : 
            for _ in range(num_random_per_dist):
 
                if edit_dist == 10 : 

                    p_select = {"add" : 1/3, "remove" : 1/3, "transform" : 1/3}
                    replace = True
                    sol_code_edited = edit_code_hoc4(solution_tokens, edit_dist, 
                        replace = replace, p_add = p_add, p_select = p_select, exact = False)
                else : 

                    p_select = {"add" : 4/10, "remove" : 4/10, "transform" : 2/10}
                    sol_code_edited = edit_code_hoc4(solution_tokens, edit_dist, p_add = p_add,  p_select = p_select)   

                random_code_df_rows.append([random_codes_idx, target_task, f'edit_{edit_dist}'])
                random_codes.append(sol_code_edited)
                random_codes_idx+=1


    random_code_df = pd.DataFrame(random_code_df_rows, columns = ["rand_id", "task", "notes"])

    path_save_info = BENCHMARK_DIR + f'/hoc4/' + 'rand_info.tsv'
    random_code_df.to_csv(path_save_info, sep = '\t', index=False)

    for code, row in zip(random_codes, random_code_df_rows) :     
        
        rand_idx, rand_task, _ = row
        path = save_dir + f'{rand_task}_rand{rand_idx}.json'
        code_ast = token_list_to_benchhmark_ast(code)

        with open(path, 'w+') as f :
            json.dump(code_ast, f, indent = 2)



if __name__ == '__main__' : 
    generate_random_codes_hoc4()