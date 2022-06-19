from code.utils.utils import * 
import numpy as np 

from .rules_obj import RulesPCFG
from code.utils.convert import ast_code_to_token_list

action_to_var = {
    "move" : "M", 
    "turn_left" : "L", 
    "turn_right" : "R"
}
action_tokens = ['turn_left','turn_right','move']

student_types_hoc4 = ['mis_1', 'mis_2', 'mis_3', 'mis_4', 'mis_5', 'mis_6']

class GenGrammarHoc4:

    def generate_grammar(self, stu_type, task_B):

        self.rules_obj = RulesPCFG()
        sol_tokens = ast_code_to_token_list(task_B["solution"])
        sol_path = list(filter(
            lambda token : token in action_tokens, 
            sol_tokens))
        sol_path = list(map(action_to_var.get, sol_path))
        sol_path = ' '.join(sol_path)

        if stu_type == 'mis_1':
            self.mis_1_generator(sol_path)
        elif stu_type == 'mis_2' : 
            self.mis_2_generator(sol_path)
        elif stu_type == 'mis_3' : 
            self.mis_3_generator(sol_path)
        elif stu_type == 'mis_4' : 
            self.mis_4_generator(sol_path)
        elif stu_type == 'mis_5' : 
            self.mis_5_generator(sol_path)
        elif stu_type == 'mis_6' : 
            self.mis_6_generator(sol_path)

        self.rules_obj.normalize()

        return self.rules_obj.rules




    def mis_1_generator(self, sol_path) : 
        '''
            confusing left/right directions when turning,
        '''
        self.rules_obj.split_terminal_sequence('S', sol_path, 'S',  start_weight = 9) 
        self.rules_obj.add_cumulative_path(sol_path, start_weight = 0.5)


        sol_tokens = np.array(sol_path.split())
        turn_idxs = np.where((sol_tokens == 'L') | (sol_tokens == 'R'))[0]
        self.rules_obj.optionally_skip_idxs(sol_path, skip_idxs = turn_idxs, start_weight = 0.5)


        self.rules_obj.add_rule( 'L', ['"turn_left"', '"turn_right"', '"move"',  'RepL RepL'], weights = [0.2, 0.6, 0.1, 0.1])
        self.rules_obj.add_rule( 'R', ['"turn_right"', '"turn_left"', '"move"',  'RepR RepR'], weights = [0.2, 0.6, 0.1, 0.1])
        self.rules_obj.add_rule( 'M', ['"move"', '"turn_left"', '"turn_right"',  'RepM RepM'],  weights = [0.7, 0.1, 0.1, 0.1])
        self.rules_obj.add_rule( 'RepM', ['RepM RepM', '"move"', '"turn_left"', '"turn_right"' ], weights = [0.4, 0.4, 0.1, 0.1])
        self.rules_obj.add_rule( 'RepL', ['RepL RepL', '"turn_left"', '"turn_right"', '"move"'],  weights = [0.4, 0.4, 0.1, 0.1])
        self.rules_obj.add_rule( 'RepR', ['RepR RepR', '"turn_right"', '"turn_left"', '"move"'],  weights = [0.4, 0.4, 0.1, 0.1])
        return self.rules_obj 

    def mis_2_generator(self,sol_path): 
        '''
            partially solving the task in terms of getting closer to the “goal”
        '''
        self.rules_obj = RulesPCFG()
        self.rules_obj.split_terminal_sequence('S', sol_path, 'S',  start_weight = 0.1)
        self.rules_obj.add_cumulative_path(sol_path, start_weight = 9.9)

        self.rules_obj.add_rule( 'L', ['"turn_left"', '"turn_right"', '"move"',  'RepL RepL'], weights = [0.7, 0.1, 0.1, 0.1])
        self.rules_obj.add_rule( 'R', ['"turn_right"', '"turn_left"', '"move"',  'RepR RepR'], weights = [0.7, 0.1, 0.1, 0.1])
        self.rules_obj.add_rule( 'M', ['"move"', '"turn_left"', '"turn_right"',  'RepM RepM'],  weights = [0.7, 0.1, 0.1, 0.1])
        self.rules_obj.add_rule( 'RepM', ['RepM RepM', '"move"', '"turn_left"', '"turn_right"' ], weights = [0.4, 0.4, 0.1, 0.1])
        self.rules_obj.add_rule( 'RepL', ['RepL RepL', '"turn_left"', '"turn_right"', '"move"'],  weights = [0.4, 0.4, 0.1, 0.1])
        self.rules_obj.add_rule( 'RepR', ['RepR RepR', '"turn_right"', '"turn_left"', '"move"'],  weights = [0.4, 0.4, 0.1, 0.1])
        return self.rules_obj 



    def mis_3_generator(self, sol_path) :  
        '''
            misunderstanding of turning functionality and writing repetitive turn commands
        '''

        self.rules_obj = RulesPCFG()
        self.rules_obj.split_terminal_sequence('S', sol_path, 'S',  start_weight = 9) 

        self.rules_obj.add_cumulative_path(sol_path, start_weight = 0.5)
        sol_tokens = np.array(sol_path.split())
        turn_idxs = np.where((sol_tokens == 'L') | (sol_tokens == 'R'))[0]
        self.rules_obj.optionally_skip_idxs(sol_path, skip_idxs = turn_idxs, start_weight = 0.5)

        self.rules_obj.add_rule( 'L', ['"turn_left"',  'TT TT', '"turn_right"'], weights = [0.05, 0.9, 0.05])
        self.rules_obj.add_rule( 'R', ['"turn_right"', 'TT TT',   '"turn_left"'], weights = [0.05, 0.9, 0.05])
        self.rules_obj.add_rule( 'M', ['"move"', '"turn_left"', '"turn_right"',  'RepM RepM'],  weights = [0.7, 0.1, 0.1, 0.1])
        self.rules_obj.add_rule( 'TT', ['TT T_TERM', '"turn_left"', '"turn_right"'])
        self.rules_obj.add_rule( 'T_TERM', ['"turn_left"', '"turn_right"'])    
        self.rules_obj.add_rule( 'RepM', ['RepM RepM', '"move"', '"turn_left"', '"turn_right"' ], weights = [0.4, 0.4, 0.1, 0.1])

        return self.rules_obj 


    def mis_4_generator(self, sol_path) :  
        '''
            adding more than the correct number of required move commands
        '''
        self.rules_obj = RulesPCFG()
        self.rules_obj.split_terminal_sequence('S', sol_path, 'S', start_weight = 9) 

        self.rules_obj.add_cumulative_path(sol_path, start_weight = 0.5)
        sol_tokens = np.array(sol_path.split())
        turn_idxs = np.where((sol_tokens == 'L') | (sol_tokens == 'R'))[0]
        self.rules_obj.optionally_skip_idxs(sol_path, skip_idxs = turn_idxs, start_weight = 0.5)

        self.rules_obj.add_rule( 'L', ['"turn_left"', '"turn_right"', '"move"',  'RepL RepL'], weights = [0.85, 0.05, 0.05, 0.05])
        self.rules_obj.add_rule( 'R', ['"turn_right"', '"turn_left"', '"move"',  'RepR RepR'], weights = [0.85, 0.05, 0.05, 0.05])
        self.rules_obj.add_rule( 'M', ['"move"', '"turn_left"', '"turn_right"',  'RepM RepM'],  weights = [0.10, 0.05, 0.05, 0.80])
        self.rules_obj.add_rule( 'RepM', ['RepM RepM', '"move"', '"turn_left"', '"turn_right"' ], weights = [0.4, 0.4, 0.1, 0.1])
        self.rules_obj.add_rule( 'RepL', ['RepL RepL', '"turn_left"', '"turn_right"', '"move"'],  weights = [0.4, 0.4, 0.1, 0.1])
        self.rules_obj.add_rule( 'RepR', ['RepR RepR', '"turn_right"', '"turn_left"', '"move"'],  weights = [0.4, 0.4, 0.1, 0.1])

        return self.rules_obj 


    def mis_5_generator(self, sol_path) :  
        '''
            forgetting to include some turns needed in the solution
        '''
        self.rules_obj = RulesPCFG()
        sol_tokens = np.array(sol_path.split())
        turn_idxs = np.where((sol_tokens == 'L') | (sol_tokens == 'R'))[0]

        self.rules_obj.optionally_skip_idxs(sol_path, skip_idxs = turn_idxs, start_weight = 1)
 
        self.rules_obj.add_rule( 'L', ['"turn_left"', '"turn_right"', '"move"',  'RepL RepL'], weights = [0.7, 0.1, 0.1, 0.1])
        self.rules_obj.add_rule( 'R', ['"turn_right"', '"turn_left"', '"move"',  'RepR RepR'], weights = [0.7, 0.1, 0.1, 0.1])
        self.rules_obj.add_rule( 'M', ['"move"', '"turn_left"', '"turn_right"',  'RepM RepM'],  weights = [0.7, 0.1, 0.1, 0.1])
        self.rules_obj.add_rule( 'RepM', ['RepM RepM', '"move"', '"turn_left"', '"turn_right"' ], weights = [0.4, 0.4, 0.1, 0.1])
        self.rules_obj.add_rule( 'RepL', ['RepL RepL', '"turn_left"', '"turn_right"', '"move"'],  weights = [0.4, 0.4, 0.1, 0.1])
        self.rules_obj.add_rule( 'RepR', ['RepR RepR', '"turn_right"', '"turn_left"', '"move"'],  weights = [0.4, 0.4, 0.1, 0.1])

        return self.rules_obj 

    def mis_6_generator(self, sol_path) : 
        '''
            attempting to randomly solve the task by adding lots of blocks
        '''
        self.rules_obj = RulesPCFG()
        self.rules_obj.split_terminal_sequence('S', sol_path, 'S', start_weight = 1) 
        self.rules_obj.add_cumulative_path(sol_path, start_weight = 1)
        sol_tokens = np.array(sol_path.split())
        turn_idxs = np.where((sol_tokens == 'L') | (sol_tokens == 'R'))[0]
        self.rules_obj.optionally_skip_idxs(sol_path, skip_idxs = turn_idxs, start_weight = 1)

        for size in range(len(sol_path)) :
            random_path_tokens = ['A' for _ in range(size)]
            random_path_str = ' '.join(random_path_tokens)
            sol_path_expanded = sol_path + ' ' + random_path_str
            self.rules_obj.split_terminal_sequence('S', sol_path_expanded, f'S_A_{size}',  start_weight = (size+1)*(size+1))

        self.rules_obj.add_rule( 'A', ['"turn_left"',  '"turn_right"', '"move"', 'A A'])
        self.rules_obj.add_rule( 'AL', ['"turn_left"',  '"turn_right"', '"move"', 'AL AL'], weights = [0.1, 0.1 , 0.5 , 0.3])
        self.rules_obj.add_rule( 'AR', ['"turn_left"',  '"turn_right"', '"move"', 'AR AR'], weights = [0.1, 0.1 , 0.5 , 0.3])
        self.rules_obj.add_rule( 'AM', ['"turn_left"',  '"turn_right"', '"move"', 'AM AM'], weights = [0.3, 0.3 , 0.1 , 0.3])
        self.rules_obj.add_rule( 'L', ['"turn_left"',  '"turn_right"', '"move"',  'AL AL'], weights = [0.1, 0.1, 0.1, 0.7])
        self.rules_obj.add_rule( 'R', ['"turn_left"',  '"turn_right"', '"move"', 'AR AR'], weights = [0.1, 0.1, 0.1, 0.7])
        self.rules_obj.add_rule( 'M', ['"turn_left"',  '"turn_right"', '"move"', 'AM AM'], weights = [0.1, 0.1, 0.1, 0.7])

        return self.rules_obj 

