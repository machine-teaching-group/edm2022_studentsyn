
import numpy as np 
from ..grammar.pcfg_general import PCFG 
from code.utils.utils import * 


class RulesPCFG:
    def __init__(self):
        self.rules = '\n'
    
    def __repr__(self):
        return self.rules

    def add_rule(self, left_handside, right_handside_list, weights = []) : 
        number_rhs = len(right_handside_list)
        if weights == [] : weights = [1.0 for _ in range(number_rhs)]

        rule = f'{left_handside} -> '
        idx = 0
        for weight, right_side in zip(weights, right_handside_list) : 
            rule += f' {right_side} [{weight}] '
            if idx != number_rhs -1 :
                rule += ' | ' 
            idx += 1

        self.rules += rule + '\n'

    def normalize(self):
        rules_str = self.rules
        pcfg_base = PCFG(rules_str)
        pcfg_base.normalize()
        rules_str_norm = pcfg_base.to_str()
        self.rules = rules_str_norm
        self.pcfg = pcfg_base

    def split_terminal_sequence(self, left_handside, sequence_str, symbol, weights = None, start_weight = None) : 

        tokens = sequence_str.split()
        transition_symbol =  f'_{symbol}{0}'
        lefthand_symbol = left_handside
        transitions = []

        if weights == None : weights = [1.0 for _ in range(len(tokens))]
        if start_weight is not None : 
            weights[0] = start_weight

        for idx, token in enumerate(tokens) : 
            if idx == len(tokens) - 2 : 
                transition_symbol = tokens[-1]
                self.add_rule(lefthand_symbol, [f'{token} {transition_symbol}'], weights = [weights[idx]])
                break 
            
            self.add_rule(lefthand_symbol, [f'{token} {transition_symbol}'], weights = [weights[idx]])
        
            lefthand_symbol = transition_symbol
            transitions.append(transition_symbol)
            transition_symbol = f'_{symbol}{idx+1}' 

        return transitions             




    
    def optionally_skip_idxs(self, sol_path,  skip_idxs, start_weight, skip_weight = 2.0, name = 'S_SKIP'):
        transition_symbols = self.split_terminal_sequence('S', sol_path, name, start_weight = start_weight)
        transition_symbols = ['S'] + transition_symbols

        check_no_consecutive(skip_idxs)

        sol_tokens = sol_path.split()
        FIRST_POS = 0
        num_toks = len(sol_tokens)
        for skip_idx in skip_idxs : 
            if skip_idx == 0 : continue 
            if skip_idx == num_toks - 1 : continue
            if skip_idx == num_toks - 2 : continue 
            if skip_idx == num_toks - 3 : 
                self.add_rule( transition_symbols[skip_idx], [f'{sol_tokens[skip_idx+1]} {sol_tokens[skip_idx+2]}'], 
                        weights = [skip_weight])
            else : 
                self.add_rule( transition_symbols[skip_idx], [f'{sol_tokens[skip_idx+1]} {transition_symbols[skip_idx+2]}'], 
                    weights = [skip_weight])

        if FIRST_POS in skip_idxs : 
            self.add_rule(transition_symbols[0], [f'{sol_tokens[1]} {transition_symbols[2]}'], 
                weights = [skip_weight*start_weight])
        
        last_idx = skip_idxs[-1]
        if (last_idx == num_toks - 2) or (last_idx == num_toks -1): 
            sol_path_mod = sol_tokens[:last_idx] + sol_tokens[last_idx+1:]
            sol_path_mod = ' '.join(sol_path_mod)
            self.optionally_skip_idxs(sol_path_mod, skip_idxs[:-1], start_weight*skip_weight, skip_weight, 
                name = f'{name}_REM_LAST')


    def add_cumulative_path(self, sol_path, start_weight):

        action_tokens = sol_path.split()
        cum_path = action_tokens[0]
        # we don't allow only single token 
        number_paths = len(action_tokens[1:-1])
        for j, action in enumerate(action_tokens[1:-1]) : 
            cum_path += ' ' + action + ' '
            self.split_terminal_sequence('S', cum_path, f'S_{j}' , start_weight = start_weight/number_paths)


    def map_sequence(self, sol_path):

        sol_tokens = sol_path.split()
        is_terminal = lambda x : x[0] == '"' and x[-1] == '"'
        tokens_mapped = []

        for token in sol_tokens :
            if is_terminal(token) : 
                if token not in self.synthetic_terminals : 
                    token_repr = token[1:-1].upper()
                    self.synthetic_terminals[token] = token_repr
                    self.add_rule(token_repr, [token])
                tokens_mapped.append(self.synthetic_terminals[token])

            else : 
                tokens_mapped.append(token)

        tokens_str = ' '.join(tokens_mapped)
        return tokens_str


    def add_condition_rules_hoc18(self, sol_path, weight_correct = 3, weight_others = 1):

        if 'C_AHEAD' in sol_path : 
            self.add_rule( 'C_AHEAD', ['"bool_path_ahead"',  '"bool_path_left"', '"bool_path_right"'], weights = [weight_correct, weight_others, weight_others])
        elif 'C_LEFT' in sol_path : 
            self.add_rule( 'C_LEFT', ['"bool_path_ahead"',  '"bool_path_left"', '"bool_path_right"'], weights = [weight_others, weight_correct, weight_others])
        elif 'C_RIGHT' in sol_path :         
            self.add_rule( 'C_RIGHT', ['"bool_path_ahead"',  '"bool_path_left"', '"bool_path_right"'], weights = [weight_others, weight_others, weight_correct])
        else : 
            print('no if condition found')
            exit(1)


def check_no_consecutive(skip_idxs):
    skip_idxs = skip_idxs.copy()
    skip_idxs.sort()
    for idx,idx_next in zip(skip_idxs, skip_idxs[1:]):
        if idx + 1 == idx_next : 
            print('not allowed, consecutive idxs')
            exit(1)
