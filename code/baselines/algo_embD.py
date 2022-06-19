from code.embeddings.embedding_util import EmbUtil
from scipy.spatial.distance import cdist 
from code.utils.convert import benchmark_code_to_student_code

import numpy as np 

class EmbD:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def observe_task(self, task_A) : 
        self.task_A = task_A
        self.emb_util = EmbUtil(task_A["name"], self.models_dir)

    def observe_stu(self, stu_A):
        self.stu_A = stu_A 

    def score(self, task_B, options):
        
        stu_A_converted = benchmark_code_to_student_code(self.stu_A)
        embedding_A = self.emb_util.get_emb([stu_A_converted])

        options_converted = [benchmark_code_to_student_code(option) for option in options]        
        option_embeddings = self.emb_util.get_emb(options_converted)
        distances = cdist(embedding_A, option_embeddings)
        choice = np.argmin(distances) 

        return choice 

    def generate(self, task_B): 
        pass 
    

def add_embD_parameters(parser, model_args_dict):
    parser.add_argument('--embD_emb_models_dir', default = 'model_1', type = str, help = 'pre-trained embedding network folder to use')

    model_args = ['embD_emb_models_dir']
    model_args_dict['embD'] = model_args

    return model_args
