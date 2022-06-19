from code.embeddings.embedding_util import EmbUtil
from code.embeddings.utils.ast_utils import json_to_ast
from scipy.spatial.distance import cdist 
from code.baselines.algo_editD import edit_dist 
from code.utils.convert import benchmark_code_to_student_code

import numpy as np 

class EditembD:

    def __init__(self, alpha, model_dir):
        self.alpha = alpha
        self.model_dir = model_dir 

    def observe_task(self, task_A) : 
        self.task_A = task_A
        self.emb_util = EmbUtil(task_A["name"], self.model_dir)

    def observe_stu(self, stu_A):
        self.stu_A = stu_A 

    def score(self, task_B, options):

        options_converted = [benchmark_code_to_student_code(option) for option in options]        
        stu_A_converted = benchmark_code_to_student_code(self.stu_A)

        tokens_A = json_to_ast(stu_A_converted).tokenize()
        embedding_A = self.emb_util.get_emb([stu_A_converted])

        option_tokens = []
        options_codes = []

        for option in options_converted : 
            tokens = json_to_ast(option).tokenize()
            option_tokens.append(tokens)
            options_codes.append(option)

        option_embeddings = self.emb_util.get_emb(options_codes)
        
        edit_distances =  [edit_dist(tokens_A, tokens_B) for tokens_B in option_tokens]
        edit_distances = np.array(edit_distances)
        emb_distances = cdist(embedding_A, option_embeddings)
        combined_dist =  emb_distances*self.alpha + edit_distances*(1-self.alpha)

        choice = np.argmin(combined_dist)
        return choice 

    def generate(self, task_B): 
        pass 

def add_editembD_parameters(parser, model_args_dict):
    parser.add_argument('--alpha', default = 0.5, type = float, help = 'convex weight between editD and embedding distance')
    parser.add_argument('--editembD_emb_models_dir', default = 'model_1', type = str, help = 'pre-trained embedding network folder to use')

    model_args = ['alpha', 'editembD_emb_models_dir']
    model_args_dict['editembD'] = model_args

    return model_args

def add_editembD_trial_args(trial):
    args_trials = {
        "alpha" : trial.suggest_categorical('alpha', [0.0, 0.25, 0.5, 0.75, 1.0])
    }
    return args_trials