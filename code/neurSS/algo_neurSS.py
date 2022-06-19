from code.utils.utils import NEURSS_DIR
import torch 
import os
import json
from copy import deepcopy

from .real_time.sample_neighbors import TrainingDataUtil
from .real_time.utils import parse_task, parse_asts, score_prob
from .real_time.train_utils import train_model_for_student, get_model_for_training

from code.neurSS.synthesizer.sampling import beam_sample
from code.neurSS.synthesizer.data import get_all_grid_rotations
from code.utils.convert import token_list_to_benchhmark_ast
from code.utils.parser.parser import Parser

import logging 

class NeurSS:

    def __init__(self, run_id, lr, 
        epochs, freeze, device, batch_size, r, emb_models_dir):


        self.run_id = run_id
        self.lr = lr
        self.epochs = epochs
        self.freeze = freeze
        self.device = device
        self.batch_size = batch_size
        self.r = r
        self.emb_models_dir = emb_models_dir

        self.stu_code_to_id = {}
        self.model_cache = {}
        self.task_cache = {}
        
    def observe_task(self, task_A) : 
        
        self.task_A = task_A
        self.source_task = task_A["name"]

        load_path = NEURSS_DIR + f'/{self.source_task}/runs/{self.run_id}/'
        logging.info(f'loading pre-trained synthesizer from {load_path}')
        with open(load_path + 'data_info.json', 'r') as f : 
            self.data_info =  json.load(f)

        model = get_model_for_training(load_path,  self.freeze)

        self.model = model 


        self.load_path = load_path 
        task_tensors = parse_task(task_A, self.data_info)
        self.inp_grid_A, self.out_grid_A, self.solution_in_A, self.solution_out_A = task_tensors
        self.dataset = TrainingDataUtil(self.source_task, self.data_info, 
            self.inp_grid_A, self.out_grid_A, self.emb_models_dir) 
        
    def observe_stu(self, stu_A): 

        self.stu_A = stu_A 
        stu_code_str = json.dumps(stu_A)

        if stu_code_str in self.stu_code_to_id : 
            stu_A_model = self.model_cache[self.stu_code_to_id[stu_code_str]]
        else :
            self.dataset.select_neighbors(stu_A, self.r)
            stu_A_model = deepcopy(self.model)
            stu_A_model = stu_A_model.to('cpu')

            train_model_for_student(
                self.epochs, self.dataset, 
                stu_A_model ,self.device, 
                self.lr, self.batch_size, self.data_info)

            stu_A_model = stu_A_model.to('cpu')
            new_id = len(self.stu_code_to_id) + 1
            self.stu_code_to_id[stu_code_str] = new_id
            self.model_cache[new_id] = stu_A_model


    def score(self, stu_A, task_B, options, return_probs = False):

        inp_grid_B, out_grid_B, solution_in_B, solution_out_B = self.get_grid_info(task_B)

        stu_code_str = json.dumps(stu_A)
        stu_A_model = self.model_cache[self.stu_code_to_id[stu_code_str]]        
        codes_in_B, codes_out_B = parse_asts(options, self.data_info)
        num_options = len(options)
        inp_grids_B = inp_grid_B.repeat(num_options, 1, 1, 1, 1)
        out_grids_B = out_grid_B.repeat(num_options, 1, 1, 1, 1)


        probs, probs_raw = score_prob(stu_A_model, 
            inp_grids_B, out_grids_B, codes_in_B, device = 'cpu')   
        stu_A_model = stu_A_model.to('cpu')
        pred = torch.argmax(probs).cpu().item()

        if return_probs :
            return pred, probs_raw
        else : 
            return pred


    def get_grid_info(self, task_B):
        if task_B["name"] in self.task_cache:
            inp_grid_B, out_grid_B, solution_in_B, solution_out_B = self.task_cache[task_B["name"]] 
        else : 
            task_tensors = parse_task(task_B, self.data_info)
            inp_grid_B, out_grid_B, solution_in_B, solution_out_B = task_tensors
            inp_grid_B = get_all_grid_rotations(inp_grid_B)
            out_grid_B = get_all_grid_rotations(out_grid_B)

            
            self.task_cache[task_B["name"]] = inp_grid_B, out_grid_B, solution_in_B, solution_out_B


        return inp_grid_B, out_grid_B, solution_in_B, solution_out_B


    def generate(self, stu_A, task_B, beam_size, top_k = 1): 
        stu_code_str = json.dumps(stu_A)
        if stu_code_str in self.stu_code_to_id:
            stu_id = self.stu_code_to_id[stu_code_str]
            stu_A_model = self.model_cache[stu_id]
        else : 
            logging.info('warning, unknown student')
            stu_A_model = self.model 
        
        parser = Parser()
        inp_grid_B, out_grid_B, solution_in_B, solution_out_B = self.get_grid_info(task_B)
        max_seq_len = 100
        get_samples = 100 

        with torch.no_grad():
            samples = beam_sample(stu_A_model, max_seq_len, inp_grid_B, out_grid_B,     
                beam_size, top_k = get_samples)

 
        codes = []
        probs = []
        for prob, code_gen_idxs in samples[0] : 
            code_tokens = list(map(
                lambda x : self.data_info["vocab"]["idx2tkn"].get(str(x)), 
                code_gen_idxs
                ))

            code_str = ' '.join(code_tokens)
        
            if parser.parse(code_str): 

                gen_ast = token_list_to_benchhmark_ast(code_tokens)
                codes.append(gen_ast)
                probs.append(torch.exp(prob).item())

                if len(codes) ==  top_k : break     

        logging.info(f'Generated {len(codes)}/{top_k} codes')
        return codes, probs 


    def save(self, save_folder):

        save_folder = save_folder + '/models/'
        os.makedirs(save_folder, exist_ok = True)

        with open(save_folder + 'stu_code_to_id.json', 'w+') as f : 
            json.dump(self.stu_code_to_id, f, indent=2) 

        id_to_stu_code = {stu_id : code_A for code_A, stu_id in self.stu_code_to_id.items()}
        for stu_id, model in self.model_cache.items() : 
            state = {
                "state_dict": model.state_dict(),
                "code_A": id_to_stu_code[stu_id],
                "load_path" : self.load_path,
                "freeze" : self.freeze,
            } 
            torch.save(state, save_folder + f'model_{stu_id}.pt')


    def load(self, load_folder):

        load_folder = load_folder + '/models/'
    
        with open(load_folder + 'stu_code_to_id.json', 'r') as f : 
            self.stu_code_to_id = json.load(f) 

        self.model_cache = {}
        for stu_id in self.stu_code_to_id.values():

            checkpoint = torch.load(load_folder + f'model_{stu_id}.pt', map_location='cpu')

            model = get_model_for_training(checkpoint["load_path"], checkpoint["freeze"])
            model.load_state_dict(checkpoint["state_dict"])

            self.model_cache[stu_id] = model

            self.load_path = checkpoint["load_path"]

def add_neurSS_parameters(parser, model_args_dict):


    parser.add_argument('--synth_run', type = str, default = 'trained_synth_1', 
    help = 'load pre-trained solution synthesizer from /data/neurSS/{source_task}/runs/{synth_run}/')
    parser.add_argument('--lr', type = float, default = 0.0001, help = 'learning rate for continual training')
    parser.add_argument('--epochs', type = int, default = 50 , help = 'number of continual training epochs')
    parser.add_argument('--freeze', type = str, default = 'encoder',
        choices = ['encoder', 'encoder_no_emb','no_emb', 'none', 'all'], help = 'part of model to freeze')
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size for continual training')
    parser.add_argument('--r', type = float, default = 1.5, help = 'radius in embedding space to sample neighbors in the student data')
    parser.add_argument('--neurSS_emb_models_dir', type = str, default = 'model_1', help = '''
        directory name to load embedding networks from in /data/embD/{emb_models_dir}/''')


    model_args = ['synth_run', 'lr', 'epochs', 'freeze', 'device', 'batch_size', 'r', 'neurSS_emb_models_dir']
    model_args_dict['neurSS'] = model_args

    return model_args

def add_neurSS_trial_args(trial):

    args_trials = {}

    args_trials["lr"] = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    args_trials["epochs"] = trial.suggest_categorical('epochs',[1] + list(range(5,61,5)))
    args_trials["r"] = trial.suggest_uniform('r',0,2)


    return args_trials