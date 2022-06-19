from code.embeddings.utils.ast_utils import json_to_ast
from code.utils.convert import benchmark_code_to_student_code
import numpy as np

class EditD:

    def __init__(self): 
        pass 

    def observe_task(self, task_A) : 
        pass 

    def observe_stu(self, stu_A):
        self.stu_A = stu_A 

    def score(self, task_B, options):

        stu_A_converted = benchmark_code_to_student_code(self.stu_A)
        tokens_A = json_to_ast(stu_A_converted).tokenize()
        
        option_tokens = []

        for option in options : 
            option_converted = benchmark_code_to_student_code(option)
            tokens = json_to_ast(option_converted).tokenize()
            option_tokens.append(tokens)

        distances = [edit_dist(tokens_A, tokens_B) for tokens_B in option_tokens]

        choice = np.argmin(distances)
        return choice 

    def generate(self, task_B): 
        pass 


def edit_dist(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

