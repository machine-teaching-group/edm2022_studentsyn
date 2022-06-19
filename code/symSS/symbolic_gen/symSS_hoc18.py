from code.utils.utils import * 
from .rules_obj import RulesPCFG
import numpy as np 
from code.utils.utils import SYMSS_DIR
from code.utils.convert import ast_code_to_token_list
import json 

action_to_var = {
    "move" : "M", 
    "turn_left" : "L", 
    "turn_right" : "R"
}
action_tokens = ['turn_left','turn_right','move']

if_tokens = [
    'IFELSE', 'ELSE', 'E(', 'E)', 'I(', 'I)',
    'C( C_AHEAD C)', 'C( C_LEFT C)', 'C( C_RIGHT C)'
    ]

while_tokens = ['WHILE C( BOOL_NO_MARKER C)','W(', 'W)'] 

student_types_hoc18 = ['mis_1', 'mis_2', 'mis_3', 'mis_4', 'mis_5', 'mis_6']

class GenGrammarHoc18:

    def generate_grammar(self, stu_type, task_B):
        
        self.rules_obj = RulesPCFG()
        self.rules_obj.synthetic_terminals = {}

        load_path = BENCHMARK_DIR + f'/hoc18/solution_basic_actions/{task_B["name"]}_code_solution_basic_actions.json'
        with open(load_path, 'r') as f :
            sol_basic_actions = json.load(f)

        sol_tokens = ast_code_to_token_list(sol_basic_actions)
        sol_path = list(filter(
            lambda token : token in action_tokens, 
            sol_tokens))
        sol_path = list(map(action_to_var.get, sol_path))
        sol_basic_actions_str = ' '.join(sol_path)


        load_path = SYMSS_DIR + f'/hoc18/distractor_paths/{task_B["name"]}_code_distractors.json'
        with open(load_path, 'r') as f :
            distractor_codes = json.load(f)

        map_dico = {
            "DEF" : '',
            "m(" : '',
            "m)" : '',
            "run" : '',
            "move" : 'M',
            "turn_left" : 'L',
            "turn_right" : 'R',
            "bool_path_ahead" : "C_AHEAD",
            "bool_path_left" : "C_LEFT",
            "bool_path_right" : "C_RIGHT"
        }
        distractor_list = []
        for code in distractor_codes:
            code = ast_code_to_token_list(code)
            code = list(map(lambda x : map_dico.get(x, '"' + x + '"' ), code))
            code = ' '.join(code).strip()
            code_mapped = self.rules_obj.map_sequence(code)
            distractor_list.append(code_mapped)

        solution_code = ast_code_to_token_list(task_B["solution"])
        solution_code = list(map(lambda x : map_dico.get(x, '"' + x + '"' ), solution_code))
        solution_code_str = ' '.join(solution_code).strip()
        solution_code_str =  self.rules_obj.map_sequence(solution_code_str)
        if stu_type == 'mis_1':
            self.mis_1_generator(solution_code_str)
        elif stu_type == 'mis_2' : 
            self.mis_2_generator(distractor_list)
        elif stu_type == 'mis_3' : 
            self.mis_3_generator(solution_code_str)
        elif stu_type == 'mis_4' : 
            self.mis_4_generator(solution_code_str)
        elif stu_type == 'mis_5' : 
            self.mis_5_generator(solution_code_str)
        elif stu_type == 'mis_6' : 
            self.mis_6_generator(sol_basic_actions_str)


        self.rules_obj.normalize()

        return self.rules_obj.rules



    def mis_1_generator(self, sol_path):
        '''
        confusing left/right directions when turning or checking conditionals
        '''
        self.rules_obj.split_terminal_sequence('S', sol_path, 'S')

        self.rules_obj.add_rule( 'L', ['"turn_left"', '"turn_right"', '"move"'], weights = [0.3, 0.6, 0.1])
        self.rules_obj.add_rule( 'R', ['"turn_right"','"turn_left"','"move"'], weights = [0.3, 0.6, 0.1])
        self.rules_obj.add_rule( 'M', ['"move"', '"turn_left"', '"turn_right"'], weights = [0.8, 0.1, 0.1])

        self.rules_obj.add_rule( 'C_LEFT', ['"bool_path_ahead"',  '"bool_path_left"', '"bool_path_right"'], weights = [1, 2, 3])
        self.rules_obj.add_rule( 'C_AHEAD', ['"bool_path_ahead"',  '"bool_path_left"', '"bool_path_right"'], weights = [3, 1, 1])
        self.rules_obj.add_rule( 'C_RIGHT', ['"bool_path_ahead"',  '"bool_path_left"', '"bool_path_right"'], weights = [1, 3, 2])


    def mis_2_generator(self, distractor_paths):
        '''
            following one of the wrong path segments
        '''
        self.rules_obj.add_rule('A_ADD',['"move"','"turn_right"','"turn_left"','A_ADD  A_ADD'], weights = [2,1,1,1,1])

        for idx, goal_path in enumerate(distractor_paths) : 

            goal_path =  add_extra_actions(goal_path)
            skip_idxs = np.where(np.array(goal_path.split()) == 'A_ADD')[0]
            self.rules_obj.optionally_skip_idxs(sol_path = goal_path, skip_idxs = skip_idxs, start_weight = 1.0,  name = f'S_{idx}' )

        self.rules_obj.add_condition_rules_hoc18(goal_path)

        self.rules_obj.add_rule( 'L', ['"turn_left"', '"turn_right"', '"move"'], weights = [0.8, 0.1, 0.1])
        self.rules_obj.add_rule( 'R', ['"turn_right"','"turn_left"','"move"'], weights = [0.8, 0.1, 0.1])
        self.rules_obj.add_rule( 'M', ['"move"', '"turn_left"', '"turn_right"'], weights = [0.8, 0.1, 0.1])





    def mis_3_generator(self, sol_path):
        '''
            misunderstanding of IfElse structure functionality and writing the same blocks in both the execution branches
        '''

        self.rules_obj.add_rule('A_ADD',['"move"','"turn_right"','"turn_left"','A_ADD  A_ADD'], weights = [2,1,1,1,1])

        sol_tokens = sol_path.split()
        if_context = False 
        else_context = False 
        if_tokens = []
        else_tokens = []
        conditionals = ['C_AHEAD', 'C_LEFT', 'C_RIGHT']
        for idx, token in enumerate(sol_tokens): 
            if token == 'I(' : if_context = True ; continue 
            if token == 'I)' : if_context = False  ; continue 
            if token == 'E(' : else_context = True ; continue 
            if token == 'E)' : else_context = False  ; continue 
            if if_context and token not in conditionals: 
                if_tokens.append( (token, idx) )
            if else_context and token not in conditionals: 
                else_tokens.append( (token, idx) )

        assert len(else_tokens) == 1
        sol_same_actions_1 = sol_tokens.copy()
        sol_same_actions_1[if_tokens[0][-1]] = else_tokens[0][0]
        num_if_tokens = len(if_tokens)
        for i in range(1, num_if_tokens):
            sol_same_actions_1[if_tokens[i][-1]] = '' 

        sol_same_actions_1_str = ' '.join(sol_same_actions_1)

        sol_same_actions_2 = sol_tokens.copy() 
        for i in range(1, num_if_tokens):
            sol_same_actions_2[if_tokens[i][-1]] = '' 

        sol_same_actions_2[else_tokens[0][-1]] = if_tokens[0][0]
        sol_same_actions_2_str = ' '.join(sol_same_actions_2)

        sol_actions_1_str =  add_extra_actions(sol_same_actions_1_str)
        skip_idxs = np.where(np.array(sol_actions_1_str.split()) == 'A_ADD')[0]
        self.rules_obj.optionally_skip_idxs(sol_path = sol_actions_1_str, skip_idxs = skip_idxs, start_weight = 1.0,  name = 'S' )

        sol_actions_2_str =  add_extra_actions(sol_same_actions_2_str)
        skip_idxs = np.where(np.array(sol_actions_2_str.split()) == 'A_ADD')[0]
        self.rules_obj.optionally_skip_idxs(sol_path = sol_actions_2_str, skip_idxs = skip_idxs, start_weight = 1.0,  name = 'Z' )
        


        self.rules_obj.add_rule( 'L', ['"turn_left"', '"turn_right"', '"move"'], weights = [0.8, 0.1, 0.1])
        self.rules_obj.add_rule( 'R', ['"turn_right"','"turn_left"','"move"'], weights = [0.8, 0.1, 0.1])
        self.rules_obj.add_rule( 'M', ['"move"', '"turn_left"', '"turn_right"'], weights = [0.8, 0.1, 0.1])

        self.rules_obj.add_condition_rules_hoc18(sol_path)

    def mis_4_generator(self, sol_path):
        '''
            ignoring the IfElse structure when solving the task
        '''
        self.rules_obj.add_rule('A_ADD',['"move"','"turn_right"','"turn_left"','A_ADD  A_ADD'], weights = [2,1,1,1,1])
        sol_str = ' '.join(sol_path.split())        
        no_if_str = sol_str
        for token in if_tokens :
            no_if_str = no_if_str.replace(token, '')


        no_if_str =  add_extra_actions(no_if_str)
        skip_idxs = np.where(np.array(no_if_str.split()) == 'A_ADD')[0]
        self.rules_obj.optionally_skip_idxs(sol_path = no_if_str, skip_idxs = skip_idxs, start_weight = 1.0,  
        name = 'S' )
        
        self.rules_obj.add_rule( 'L', ['"turn_left"', '"turn_right"', '"move"'], weights = [0.8, 0.1, 0.1])
        self.rules_obj.add_rule( 'R', ['"turn_right"','"turn_left"','"move"'], weights = [0.8, 0.1, 0.1])
        self.rules_obj.add_rule( 'M', ['"move"', '"turn_left"', '"turn_right"'], weights = [0.8, 0.1, 0.1])


    def mis_5_generator(self, sol_path): 
        '''
            ignoring the While structure when solving the task    
        '''

        self.rules_obj.add_rule('A_ADD',['"move"','"turn_right"','"turn_left"','A_ADD  A_ADD'], weights = [2,1,1,1,1])
        sol_str = ' '.join(sol_path.split())        
        no_while_str = sol_str
        for token in while_tokens :
            no_while_str = no_while_str.replace(token, '')

        no_while_str =  add_extra_actions(no_while_str)
        no_while_str = no_while_str.replace('IFELSE', 'A_ADD IFELSE')
        no_while_str = no_while_str.replace('E)', 'E) A_ADD')

        skip_idxs = np.where(np.array(no_while_str.split()) == 'A_ADD')[0]
        self.rules_obj.optionally_skip_idxs(sol_path = no_while_str, skip_idxs = skip_idxs, start_weight = 1.0,  
        name = 'S' )

        self.rules_obj.add_rule( 'L', ['"turn_left"', '"turn_right"', '"move"'], weights = [0.8, 0.1, 0.1])
        self.rules_obj.add_rule( 'R', ['"turn_right"','"turn_left"','"move"'], weights = [0.8, 0.1, 0.1])
        self.rules_obj.add_rule( 'M', ['"move"', '"turn_left"', '"turn_right"'], weights = [0.8, 0.1, 0.1])

        self.rules_obj.add_condition_rules_hoc18(sol_path)


    def mis_6_generator(self, sol_path_basic_actions) :
        '''
            attempting to solve the task by using only the basic action blocks in {turnLeft, turnRight, move}
        '''

        self.rules_obj.split_terminal_sequence('S', sol_path_basic_actions, 'S')
        action_tokens = sol_path_basic_actions.split()
        cum_path = action_tokens[0]
        for j, action in enumerate(action_tokens[1:-1]) : 
            cum_path += ' ' + action + ' '
            self.rules_obj.split_terminal_sequence('S', cum_path, f'S_{j}' )


        self.rules_obj.add_rule( 'A', ['"turn_left"',  '"turn_right"', '"move"', 'A A'], weights = [0.3, 0.3 , 0.3 , 0.1])
        self.rules_obj.add_rule( 'L', ['"turn_left"',  '"turn_right"', '"move"',  'A A'], weights = [0.7, 0.1, 0.1, 0.1])
        self.rules_obj.add_rule( 'R', ['"turn_left"',  '"turn_right"', '"move"', 'A A'], weights = [0.1, 0.7, 0.1, 0.1])
        self.rules_obj.add_rule( 'M', ['"turn_left"',  '"turn_right"', '"move"', 'A A'], weights = [0.1, 0.1, 0.7, 0.1])

    


def add_extra_actions(sol_path):
    sol_path = sol_path.replace('WHILE','A_ADD WHILE')
    sol_path = sol_path.replace('W(','W( A_ADD')
    sol_path = sol_path.replace('I(','I( A_ADD')
    sol_path = sol_path.replace('E(','E( A_ADD')
    sol_path = sol_path.replace('I)','A_ADD I)')
    sol_path = sol_path.replace('E)','A_ADD E)')
    sol_path = sol_path.replace('W)','A_ADD W)')
    return sol_path    
