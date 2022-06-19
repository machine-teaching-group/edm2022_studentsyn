import numpy as np 
from code.utils.utils import * 
from types import SimpleNamespace
import re 

class CodeGenerator:

    def __init__(self):
        self.token_names, self.prodrules =  Rules().get()
        self.non_terminals = list(self.prodrules.keys())
        self.action_functions = ['move','turn_right','turn_left']
        self.action_tokens = [self.token_names.t_MOVE,self.token_names.t_TURN_LEFT,self.token_names.t_TURN_RIGHT]
        self.action_generating_blocks = ['action_one','action_block','inner_stmt_stmt']    
        self.parse_rules()
 
    # create prodrules table
    def parse_rules(self):
        self.tokens = {}
        self.prodrules_parsed = {}

        for k,v in self.prodrules.items():
            if not isinstance(v,str):
                rules = [x.split() for x in v[0].split('|')]
                weights = v[1]
            else : 
                rules = [x.split() for x in v.split('|')]
                weights = [1 for _ in range(len(rules))]
            probs = np.array(weights) / np.sum(weights)
            self.prodrules_parsed[k] = (rules,probs)

    def assert_min_moves(self,min_move,code):

        # if we have less than min number of `move`        
        count_diff = min_move - code.count(self.token_names.t_MOVE)

        # replace some random tokens with `move`
        if count_diff > 0:
            action_candidates = []
            tokens = code

            for idx, token in enumerate(tokens):
                if token in self.action_functions and token != self.token_names.t_MOVE:
                    action_candidates.append(idx)

            idxes = self.rng.choice(
                    action_candidates, min(len(action_candidates), count_diff))

            for idx in idxes:
                tokens[idx] = self.token_names.t_MOVE

    def prune_code(self,code):

        '''
            fix redundant code like
            turn_left turn_right --> ''
        '''
        code_str = ' '.join(code)
        tl = self.token_names.t_TURN_LEFT
        tr = self.token_names.t_TURN_RIGHT
        old_str = ''
        flip = True 
        while old_str != code_str and flip :
            old_str = code_str 
            code_str = re.sub(f'({tr} {tl}|{tl} {tr})*','',code_str)
            code_str = re.sub(f'{tl} {tl} {tl}',f'{tr}',code_str)
            code_str = re.sub(f'{tr} {tr} {tr}',f'{tl}',code_str)
            flip = not flip 
        l_braces = [   
            self.token_names.t_E_LBRACE,
            self.token_names.t_M_LBRACE,        
            self.token_names.t_W_LBRACE,
            self.token_names.t_I_LBRACE        
        ]


        # empty braces are a syntactic error, fix
        for sym in l_braces : 
            pat = f'{sym[0]}\(\s*{sym[0]}\)'
            code_str = re.sub(pat,f'{sym[0]}( {self.rng.choice(self.action_tokens)} {sym[0]})',code_str)
        return code_str.split()

    # ------ Create random code  ---------
    def random_code(self,rng, min_move = 1 , task = 'hoc18'):
        self.rng = rng 
        if task == 'hoc4':
            # choose by fixed length 
            # and not probability of termination for now
            # like original code
            start_token = 'prog_basic'
            lens = np.arange(2,13)
            len_weights = np.arange(20 + 2,20 + 13)            
            self.max_actions = self.rng.choice(lens, p = len_weights / np.sum(len_weights) )
        else : 
            start_token = 'prog'
            # maximum consequent block of actions allowed 
            self.max_actions = 4

            
        self.action_count = 0 
        code = self.random_tokens(start_token)
        self.assert_min_moves(min_move,code)
        code = self.prune_code(code)

        return code

    def random_tokens(self, start_token="prog"):


            codes = []
            candidates,probs = self.prodrules_parsed[start_token]
            prod = candidates[self.rng.choice(len(candidates), p = probs)]

            for term in prod:

                # if non terminal recursively extend
                if term in self.action_generating_blocks :
                    if self.action_count > self.max_actions : 
                        self.action_count = 0 
                        continue 

                if term in self.non_terminals:
                    expanded = self.random_tokens(term)

                    # no loop without move
                    if term == 'in_loop':
                        self.assert_min_moves(1,expanded) 

                    codes.extend(expanded)

                # else replace with code token
                else: 
                    if term == 'INT': token = self.random_INT()
                    else : token = getattr(self.token_names,f't_{term}')
                    codes.append(str(token).replace('\\', ''))
                    if token in self.action_functions : 
                        self.action_count += 1 
                    else : 
                        self.action_count = 0 
            
            return codes    




class Rules:
    # parser tokens
    token_names = SimpleNamespace(
        t_M_LBRACE = 'm\(',
        t_M_RBRACE = 'm\)',
        t_C_LBRACE = 'c\(',
        t_C_RBRACE = 'c\)',
        t_W_LBRACE = 'w\(',
        t_W_RBRACE = 'w\)',
        t_I_LBRACE = 'i\(',
        t_I_RBRACE = 'i\)',
        t_E_LBRACE = 'e\(',
        t_E_RBRACE = 'e\)',
        t_RUN = 'run',
        t_DEF = 'DEF',
        t_WHILE = 'WHILE',
        t_IFELSE = 'IFELSE',
        t_ELSE = 'ELSE',
        t_PATH_AHEAD = 'bool_path_ahead',
        t_PATH_LEFT = 'bool_path_left',
        t_PATH_RIGHT = 'bool_path_right',
        t_MOVE = 'move',
        t_TURN_RIGHT = 'turn_right',
        t_TURN_LEFT = 'turn_left',
        t_NO_MARKER = 'bool_no_marker'
    )   
    prodrules = {
    


    # -------- basic actions only ---------------
    # we complicated generation to get better qualities program 

    'prog_basic' : 'DEF RUN M_LBRACE init_turn MOVE inner_stmt_stmt M_RBRACE',

    'init_turn' : ('''
          TURN_RIGHT 
        | TURN_LEFT 
        | TURN_RIGHT TURN_RIGHT 
        | TURN_LEFT TURN_LEFT 
        |
        ''',[1,1,1,1,2]),

    'inner_stmt_stmt' : 
    ('''
        inner_basic inner_stmt_stmt 
        | 
    ''',[1,0]
    ),
    'inner_basic' : ('''
        MOVE 
        | TURN_RIGHT MOVE 
        | TURN_LEFT MOVE 
    
    ''',[5,2,2]),


    # ----------- while structure programs  -------------

    'prog' :  'DEF RUN M_LBRACE action_block top M_RBRACE',
    
    'top'  :  ''' while 
                ''',
    'action_block' : ('action action_block | ',[3,2]),
    'action_one' : ('action action_block | action',[3,2]),

    'while' : 'action_block WHILE C_LBRACE NO_MARKER C_RBRACE W_LBRACE in_loop W_RBRACE',
        
    'ifelse' : '''
            IFELSE C_LBRACE PATH_AHEAD C_RBRACE I_LBRACE action_one I_RBRACE ELSE E_LBRACE action_one E_RBRACE
        |   IFELSE C_LBRACE PATH_LEFT C_RBRACE I_LBRACE  action_one I_RBRACE ELSE E_LBRACE action_one E_RBRACE
        |   IFELSE C_LBRACE PATH_RIGHT C_RBRACE I_LBRACE action_one I_RBRACE ELSE E_LBRACE action_one E_RBRACE
        ''',
    
    'cond' :    '''   PATH_RIGHT
                    | PATH_AHEAD 
                    | PATH_LEFT
                ''',
        
        'in_loop' :   
                    ('''
                     action_block ifelse action_block 
                    | action_block ifelse action_block ifelse action_block
                    | action_one action_one
                    ''',[1,1,2]),
        'action' : ('''MOVE 
                    | TURN_RIGHT 
                    | TURN_LEFT ''',[1,1,1])
    }

    def get(self):
        return self.token_names, self.prodrules


