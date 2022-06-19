import os 
import json
import argparse
from code.utils.utils import OUTPUT_DIR
from code.exp_generative.generate import generate_main, add_args_generate
from code.exp_discriminative.single_run import add_training_arguments, single_run_main
from copy import deepcopy

from code.baselines.algo_tutorSS import add_tutorSS_parameters
from code.neurSS.algo_neurSS import add_neurSS_parameters

def run_generation(args):
    benchmark_file = args.benchmark_file
    benchmark_samples = args.benchmark_samples
    num_trials = args.num_trials

    source_task = args.source_task
    load_run = args.load_run
    top_k = args.top_k

    if load_run is None : 
        load_run = f'{benchmark_file}_neurSS_{source_task}'

    run_path = OUTPUT_DIR + f'/discriminative/{load_run}/'


    parser_single_run = argparse.ArgumentParser()
    model_args_dict = {}
    add_training_arguments(parser_single_run)
    add_neurSS_parameters(parser_single_run, model_args_dict)    
    add_tutorSS_parameters(parser_single_run, model_args_dict)    
    default_single_runs = parser_single_run.parse_args(['--model_name', 'neurSS'])



    # if load_run not specified or doesn't exist, train a new neurSS model
    # by optimizing over benchmark
    if not os.path.exists(run_path):

        run_name = load_run

        default_single_runs.run_name = run_name
        default_single_runs.benchmark_file = benchmark_file
        default_single_runs.model_name = 'neurSS'
        default_single_runs.source_task = source_task
        default_single_runs.benchmark_samples = benchmark_samples
        default_single_runs.num_trials = num_trials
        default_single_runs.save_stud_models = True 
        default_single_runs.optimize = True
        
        single_run_main(default_single_runs, model_args_dict)        
        
        synth_name = run_name

    else : 
        synth_name = load_run 

    model_args_dict = {}
    parser_generate = argparse.ArgumentParser()
    add_args_generate(parser_generate)
    args_generate = parser_generate.parse_args(['--model_name', 'neurSS'])    
    args_generate.source_task = source_task

    for model in ['neurSS','symSS','tutorSS']: 

        save_name = f'generated_codes_{source_task}/generated_codes_{model}'
        cmd_args = deepcopy(args_generate)
        cmd_args.model_name = model
        cmd_args.save_name = save_name 
        cmd_args.top_k = top_k
        
        if model == 'neurSS' : 
            cmd_args.load_run = synth_name

        generate_main(cmd_args, model_args_dict)

        print(f'Saving results at {OUTPUT_DIR}/generative/{save_name}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_task", default = 'hoc4')

    parser.add_argument("--load_run", default = None, type = str,
        help = '''run name in outputs/discriminative/ to load pre-trained neurSS student model. If it is not specified or 
        cannot be found we train a new neurSS model to use for generation''')

    parser.add_argument('--benchmark_file', type = str, default = 'generate', help = 'benchmark name used for training the neurSS model')
    parser.add_argument('--benchmark_samples', type = int,  default = 20, help = 'number benchmark samples usef for training the neurSS model')
    parser.add_argument('--num_trials', type = int, default = 20, help = 'number of optimization trials over benchmark for the neurSS model')
    parser.add_argument('--top_k', type = int, default = 3, help = 'number of top-scoring codes to generate')

    args = parser.parse_args()

    run_generation(args)