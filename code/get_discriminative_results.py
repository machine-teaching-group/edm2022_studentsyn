import re
import time 
import os 
import argparse 
from code.utils.utils import OUTPUT_DIR, EMB_DIR, NEURSS_DIR
from code.exp_discriminative.single_run import single_run_main, add_training_arguments

from code.baselines.algo_editembD import add_editembD_parameters
from code.neurSS.algo_neurSS import add_neurSS_parameters
from code.baselines.algo_tutorSS import add_tutorSS_parameters

import numpy as np 
from copy import deepcopy
import json 


def exists_embedding(embedding_models, source_task):
    for emb_model in embedding_models:
        save_dir = EMB_DIR + f'/{source_task}/{emb_model}/'
        if not os.path.exists(save_dir + f'{source_task}_best.pt'):
            print(save_dir + f'{source_task}_best.pt')
            print('embedding model does not exist!')
            raise Exception('path does not exist')

        if not os.path.exists(save_dir + f'{source_task}_data_embedded.p'):
            print(save_dir +  f'{source_task}_data_embedded.p')
            print('student embeddings do not exist!')
            raise Exception('path does not exist')
    

def exists_synthesizer(synthesizer_models, source_task):
    for synth_model in synthesizer_models:
        path = NEURSS_DIR + f'/{source_task}/runs/{synth_model}/model_best.pt'
        if not os.path.exists(path):  
            print(path)
            print('synthesizer model does not exist!')      
            raise Exception('path does not exist')


def run_multiple_runs(params_all_models, model, model_save , args_general, args_loop):
    run_prefix, default_args_single_run, benchmark_file, model_args_dict, results = params_all_models
    run_name = run_prefix + f'{model_save}/'
    cmd_args = deepcopy(default_args_single_run)
    cmd_args.model_name = model 

    for arg, value in args_general.items() : 
        setattr(cmd_args, arg, value)

    accuracies = []    
    for idx in range(1, args.num_repeat + 1):

        cmd_args.benchmark_file = benchmark_file + f'_CV_{idx}'
        cmd_args.run_name = run_name + f'CV_{idx}'

        for arg, sequence in args_loop.items():
            setattr(cmd_args,arg,sequence[idx-1])

        ret_acc = single_run_main(cmd_args, model_args_dict)
        accuracies.append(ret_acc)

    results[model_save] = {
        "accuracies": accuracies, 
        "mean test acc" : np.mean(accuracies)
        }



def run_final_results(args):
    t1 = time.time()


    results = {}

    source_task = args.source_task
    benchmark_file = args.benchmark_file
    num_repeat = args.num_repeat
    benchmark_samples = args.benchmark_samples
    num_trials = args.num_trials
    num_folds = args.num_folds

    synthesizer_models = args.synthesizer_models.split(',')
    embedding_models = args.embedding_models.split(',')

    exists_synthesizer(synthesizer_models, source_task)
    exists_embedding(embedding_models, source_task)
    assert len(embedding_models) >= num_repeat
    assert len(synthesizer_models) >= num_repeat

    results['source_task'] = source_task
    results['benchmark_file'] = benchmark_file
    results['num_repeat'] = num_repeat 
    results['benchmark_samples'] = benchmark_samples
    results['num_trials'] = num_trials

    parser_single_run = argparse.ArgumentParser()
    add_training_arguments(parser_single_run)
    model_args_dict = {}
    add_editembD_parameters(parser_single_run, model_args_dict)    
    add_neurSS_parameters(parser_single_run, model_args_dict)    
    add_tutorSS_parameters(parser_single_run, model_args_dict)    
    # some values are required for parsing, we will override later
    default_args_single_run = parser_single_run.parse_args(
        ['--model_name', 'randD']
        )
    default_args_single_run.benchmark_samples = benchmark_samples
    default_args_single_run.cross_validate = True
    default_args_single_run.source_task = source_task
    default_args_single_run.num_folds = num_folds 

    run_prefix =  f'/evaluation_{benchmark_file}_{source_task}/'

    print('Evaluating baselines')

    results = {}
    # evaluate randD
    print('randD')    
    params_all_models = (run_prefix, default_args_single_run, benchmark_file, model_args_dict, results)
    run_multiple_runs(params_all_models, 'randD', 'randD', {}, {})

    print('editEmbD')
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        run_multiple_runs(params_all_models, 'editembD', f'editembD_alpha_{alpha}', {'alpha' : alpha }, {'editembD_emb_models_dir' : embedding_models})        


    # evaluate techniques
    run_multiple_runs(params_all_models, 'tutorSS', 'tutorSS', {}, {})        
    run_multiple_runs(params_all_models, 'symSS', 'symSS', {}, {})        


    run_multiple_runs(params_all_models, 'editembD', 'editembD_cross_val', {'optimize' : True, 'num_trials' : num_trials}, {'editembD_emb_models_dir' : embedding_models})        
    run_multiple_runs(params_all_models, 'neurSS', 'neurSS', {'optimize' : True, 'num_trials' : num_trials}, {'neurSS_emb_models_dir' : embedding_models, 'synth_run' : synthesizer_models})        

    # calculate best of baselines 
    baselines = []
    for model, acc in results.items():
        if 'editembD_alpha' in model: 
            baselines.append([acc['mean test acc'],acc])

    results['editD'] = results['editembD_alpha_0.0']
    results['editembD'] = max(baselines, key = lambda x : x[0])[1]


    t2 = time.time()
    print(f'Evaluation done after {t2-t1} seconds')

    save_path = OUTPUT_DIR + f'/discriminative/{run_prefix}/final_results.json'
    print(f'Saving results at : {save_path}')
    with open(save_path, 'w+') as f :
        json.dump(results, f, indent=2)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--benchmark_file', default = 'RESULTS', help ='''
         save/load a set of generated benchmark instances from /outputs/saved_benchmarks/{benchmark_file}''',
        type = str)    
    parser.add_argument('--benchmark_samples', default = 20, help = '''
         number of full benchmark instances to sample, the final benchmark will contains benchmark_samples * 18 instances.''', type = int)
    parser.add_argument('--source_task', default = 'hoc4', choices = ['hoc4','hoc18'])
    parser.add_argument('--num_trials', default = 20, help = 'number of trials in optimization for parameter selection', type = int) 
    parser.add_argument("--num_folds", default = 10, type = int, help = 'number of folds used for cross validation')


    parser.add_argument('--num_repeat', default = 1, help = '''number of independent cross-validations to perform
        each with a synthesizer model in `--synthesizer_models`, and an embedding model in `--embedding_models`. ''', type = int)
    parser.add_argument('--embedding_models', default = 'model_1', help = '''embedding models to use for editembD and neurSS neighbor selection.  
        Add more models by seperating with  `,`, e.g. --embedding_models model_1,model_2,model_3
        ''', type = str) 
    parser.add_argument('--synthesizer_models', default = 'trained_synth_1', help = '''
        pre-trained synthesizer models to use for neurSS. Add more models by seperating with  `,`, e.g. --synthesizer_models trained_synth_1,trained_synth_2,trained_synth_3''', type = str) 


    args = parser.parse_args()
    run_final_results(args)
