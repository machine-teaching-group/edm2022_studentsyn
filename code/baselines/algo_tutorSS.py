from code.utils.utils import TUTOR_DIR
import pandas as pd 
from code.benchmark.score import score 
import json 

class TutorSS:

    def observe_task(self, task_A): 
        self.task_A_name = task_A["name"]

    def get_accuracy(self, tutor_id = None):
        base_path = TUTOR_DIR + f'/{self.task_A_name}/'
        codes =  pd.read_csv(base_path + f'codes_{self.task_A_name}.tsv', sep = '\t')
        responses =  pd.read_csv(base_path + f'res_discriminative_{self.task_A_name}.tsv', sep = '\t')

        if tutor_id is not None : 
            tutors = [tutor_id]
        else : 
            tutors = ['TutorSS1','TutorSS2','TutorSS3']

        accuracy_per_tutor = []

        for tutor in tutors:

            responses_tutor = responses[responses.tutor == tutor]
            instances = []
            codesB = [f'codeB{alpha}' for alpha in 'abcdefghij']

            for idx in range(len(responses_tutor)): 

                instance = responses_tutor.iloc[idx]
                options = codes.iloc[ instance[codesB]].code.values
                code_A = codes.iloc[ instance['codeA'] ].code
                instances.append([None, code_A, None, options])

            labels = responses_tutor.label.values
            preds = responses_tutor.pred.values
            accuracy = score(instances, labels, preds)
            accuracy_per_tutor.append(accuracy)
        
        mean_accuracy = sum(accuracy_per_tutor) / len(accuracy_per_tutor)
        return mean_accuracy, preds 


    def generate(self, task_B_name, stud_A_id, tutor_id = None): 

        base_path = TUTOR_DIR + f'/{self.task_A_name}/'
        responses_path = base_path + f'res_generative_{self.task_A_name}.json'
        with open(responses_path, 'r') as f : 
            responses = json.load(f)

        df_dict = {}
        for i,res in enumerate(responses, start = 1) : 
            df_dict[i] = res

        responses_generative = pd.DataFrame.from_dict(df_dict, orient = 'index')
        codes =  pd.read_csv(base_path + f'codes_{self.task_A_name}.tsv', sep = '\t')
        gen_codes = []

        if tutor_id is not None : 
            tutors = [tutor_id]
        else : 
            tutors = ['TutorSS1','TutorSS2','TutorSS3']

        for tutor in tutors : 

            responses_generative_tutor =  responses_generative[responses_generative.tutor == tutor]
            responses_generative_tutor =  responses_generative_tutor[responses_generative_tutor.target_task == task_B_name]
            selected_idx = codes[(codes.stu_id == stud_A_id) & (codes.task == self.task_A_name)]
            selected_idx = selected_idx.code_idx.item()

            if int(selected_idx) in responses_generative_tutor.codeA.tolist():            
                code_ast = responses_generative_tutor[responses_generative_tutor.codeA == int(selected_idx)].code.item()
                gen_codes.append(code_ast)        
            else : 
                gen_codes.append({"type" : "empty"})        

        return gen_codes

def add_tutorSS_parameters(parser, model_args_dict):
    parser.add_argument('--tutor_id', type = str, default = None, help = 'specify tutor id',
        choices = ['TutorSS1','TutorSS2','TutorSS3'])
    model_args = ['tutor_id']
    model_args_dict['tutorSS'] = model_args
    return model_args