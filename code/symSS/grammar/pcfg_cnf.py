from code.utils.utils import * 
from code.utils.convert import ast_code_to_token_list

from .pcfg_general import PCFG

import json 
import numpy as np

CYTHON = True

if CYTHON : 
    import pyximport
    pyximport.install(setup_args={'include_dirs': np.get_include()}, language_level = 3)
    from .algorithms_cython import inside_algorithm, viterbi
else : 
    from .algorithms_python import inside_algorithm, viterbi


class GrammarCNF:
    ''' 
        PCFG grammar object       

        grammar is required to be in CNF : 
            rules are only in the form : 
                A -> B C (binary)
                A -> a (unary, `a` is terminal)        
            no epsilon transitions allowed
    '''
    def __init__(self, grammar_str, terminals):
        self.grammar = PCFG(grammar_str)
        unary_rules = []
        binary_rules = []

        symbols = set()
        for prod in self.grammar.productions():

            L = prod.left_side
            R = prod.right_side

            symbols.add(L)
            prob = prod.prob

            unary = len(R) == 1
            
            if unary : 
                terminal = R[0].strip('"').strip("'")
                unary_rules.append([L, terminal, prob])
            else : 
                for r in R : 
                    symbols.add(r)        
                binary_rules.append([L, R[0], R[1], prob])
    

        unary_rules = np.array(unary_rules)
        binary_rules = np.array(binary_rules)

        symbols.remove('S')
        symbols = list(symbols)

        symbols.sort()

        terminals.sort()
        variable_to_idx = {'S' : 0}
        for idx, var in enumerate(symbols):
            variable_to_idx[var] = idx + 1

        idx_to_terminal = {idx : terminal for idx, terminal in enumerate(terminals)}    
        terminal_to_idx = {terminal : idx for idx, terminal in idx_to_terminal.items()}

        for i in range(len(unary_rules)) : 
            unary_rules[i,1] = terminal_to_idx[unary_rules[i,1]]
    
        for var, idx in variable_to_idx.items():
            binary_rules[binary_rules == var] = idx
            unary_rules[unary_rules == var] = idx

        self.binary_rules = np.array(binary_rules, dtype = float) 
        self.unary_rules = np.array(unary_rules, dtype = float) 

        self.num_variables = len(variable_to_idx)

        self.variable_to_idx = variable_to_idx
        self.idx_to_variable = {v : k for k,v in variable_to_idx.items()}
        self.idx_to_terminal = idx_to_terminal
        self.terminal_to_idx = terminal_to_idx

        self.binary_to_idx = { self.idx_to_variable[b[0]] + self.idx_to_variable[b[1]] + self.idx_to_variable[b[2]] : i
            for i,b in enumerate(self.binary_rules)}

        self.unary_to_idx = { self.idx_to_variable[u[0]] + self.idx_to_terminal[u[1]] : i
            for i,u in enumerate(self.unary_rules)}

        self.inverse_binary_to_id = { v : k for k, v in self.binary_to_idx.items()}
        self.inverse_unary_to_id =  { v : k for k, v in self.unary_to_idx.items()}
    

    def generate_topk(self,top_k = 2, iterations = 100):
        return self.grammar.generate_topk(top_k, iterations)

    def find_bin_rule(self, str) : 
        return self.binary_rules[self.binary_to_idx[str]]

    def find_un_rule(self, str) : 
        return self.unary_rules[self.unary_to_idx[str]]


    def parse_code_ast(self, code_ast):
        code_tokens = ast_code_to_token_list(code_ast)
        clear_tokens = set(['<s>','m(','m)','<pad>','DEF','run'])
        code_valuable_tokens = list(filter(
            lambda token : token not in clear_tokens, code_tokens))
        code_grammar_idxs = list(map(
            lambda token : self.terminal_to_idx.get(token, token), 
            code_valuable_tokens))

        return code_grammar_idxs

    def score(self, seq, method = 'viterbi'):
        '''
            score using the PCFG model in different ways, 

                1) with CYK parser, e.g. finding \sum_t p(t|seq), sum prob of all the trees 
                2) with Viterbi                  \max_t p(t|seq), prob of max tree
        
            return : 
                - prob, len
        '''

        if isinstance(seq, dict):
            seq = self.parse_code_ast(seq)
        else : 
            seq = list(map(
                lambda token : self.terminal_to_idx.get(token, token), 
                seq))
 
        if method == 'viterbi':
            V,max_len, left, right, split = viterbi(np.array(seq), self.num_variables, 
                self.unary_rules, self.binary_rules)
            return V[0,0,-1], max_len[0,0,-1]
        else : 
            A = inside_algorithm(np.array(seq), self.num_variables, 
                self.unary_rules, self.binary_rules)
            return A[0,0,-1], len(seq)


    def parse(self, seq) : 
        '''
            similar to score, return the CYK parsing table     
        '''
        if isinstance(seq, dict):
            seq = self.parse_code_ast(seq)
        else : 
            seq = list(map(
                lambda token : self.terminal_to_idx.get(token, token), 
                seq))
        ret = viterbi(np.array(seq), self.num_variables, 
                self.unary_rules, self.binary_rules)
        return ret[0]


    def print_tree(self, seq):
        '''
            print the json tree of parsing for this string         
        '''
        if isinstance(seq, dict):
            seq = self.parse_code_ast(seq)
        else : 
            seq = list(map(
                lambda token : self.terminal_to_idx.get(token, token), 
                seq))

        inside_probs, max_len, left, right, split  = viterbi(np.array(seq), self.num_variables, 
                    self.unary_rules, self.binary_rules)

        if inside_probs[0,0,-1] == 0 : 
            print('not parsed')
            return 

        ret = self._rec_construct_tree(0, 0, -1 , split, left ,right, inside_probs)

        print(json.dumps(ret,indent=1))


    def _rec_construct_tree(self,root, l_idx, r_idx, 
        split, left , right, inside_probs):
        '''
            aux function, 
            iteratively construct parsing tree        
        '''
        spl = int(split[root,l_idx, r_idx])
        l_term = int(left[root, l_idx, r_idx])
        r_term = int(right[root, l_idx, r_idx])
        prob = inside_probs[root,l_idx,r_idx]
        if spl == - 1 : 
            return  {   "rule" : f'{self.idx_to_variable[root]} -> {self.idx_to_terminal[l_term]}',
                        "prob"  : prob,
                        "children" : []
                    }

        rule = f'{self.idx_to_variable[root]} -> {self.idx_to_variable[l_term]} {self.idx_to_variable[r_term]}'
        root_l = int(left[root,l_idx, r_idx])
        root_r = int(right[root,l_idx, r_idx])

        ch1 = self._rec_construct_tree(root_l,l_idx, spl, split, left , right, inside_probs)
        ch2 = self._rec_construct_tree(root_r,spl+1, r_idx, split, left , right, inside_probs)
        
        return  {   "rule" : rule ,
                    "prob"  : prob, 
                    "children" : [ ch1, ch2 ]
                }

    def show(self,nt = None):
        '''
            show rules of PCFG 
            if nt is specified show rules for a specific non-terminal
        '''

        for u in self.unary_rules : 
            if nt is not None and self.idx_to_variable[int(u[0])] != nt : continue
            print(f'{self.idx_to_variable[int(u[0])]} -> {self.idx_to_terminal[u[1]] }  [{round(u[-1]*100,1)}]')

        print('\n') 
        for b in self.binary_rules : 
            if nt is not None and self.idx_to_variable[int(b[0])] != nt : continue
            print(f'{self.idx_to_variable[int(b[0])]} -> {self.idx_to_variable[int(b[1])]}  {self.idx_to_variable[int(b[2])]} [{round(b[-1]*100,1)}]')

    def tostr(self):
        grammar_str = ''
        for unary_rule in self.unary_rules : 
            L, R, prob = unary_rule
            L = self.idx_to_variable[int(L)]
            R = self.idx_to_terminal[R] 
            prob = round(prob*100,1)
            grammar_str += f'{L} -> \"{R}\" [{prob}]\n'

        for binary_rule in self.binary_rules  : 
            L, R1, R2, prob = binary_rule
            L = self.idx_to_variable[int(L)]
            R1 = self.idx_to_variable[R1] 
            R2 = self.idx_to_variable[R2] 
            prob = round(prob*100,1)
            grammar_str += f'{L} -> {R1} {R2} [{prob}]\n'

        return grammar_str     

    def __repr__(self) : 
        return self.tostr()



if __name__ == '__main__':
    grammar_str = '''
        S ->  A S [0.2] | B S  [0.4] | "a" [0.2] | "b" [0.2]
        A -> "a"
        B -> "b"    
    '''
    grammar = GrammarCNF(grammar_str, ["a","b"])
    print(grammar)
    print(grammar.tostr())
    grammar.show()
    seq = ['a', 'b', 'b' , 'b', 'a']
    ret = grammar.parse(seq)
    print(ret)
    grammar.print_tree(seq)
    print(grammar.score(seq, 'viterbi'))
    print(grammar.score(seq, 'CYK'))
    print(grammar.generate_topk(top_k = 3))
    print(grammar.find_bin_rule('SAS'))
    print(grammar.find_un_rule('Aa'))