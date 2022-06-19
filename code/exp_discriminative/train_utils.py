from code.baselines.algo_editD import EditD 
from code.baselines.algo_editembD import EditembD 
from code.baselines.algo_randD import RandD 
from code.baselines.algo_embD import EmbD 
from code.baselines.algo_tutorSS import TutorSS 
from code.neurSS.algo_neurSS import NeurSS 
from code.symSS.algo_symSS import SymSS

from code.benchmark.benchmark import Benchmark
from code.benchmark.score import score 
from code.utils.utils import OUTPUT_DIR

import json 
import numpy as np 
import pickle 
import os 
from types import SimpleNamespace
import pandas as pd 
import time 
import logging
import optuna 
from code.baselines.algo_editembD import add_editembD_trial_args

from code.neurSS.algo_neurSS import add_neurSS_trial_args
from copy import deepcopy


def select_model(model_name, model_args):
    if model_name == 'editD':
        model = EditD()
    elif model_name == 'randD':  
        model = RandD()
    elif model_name == 'embD':  
        model = EmbD(model_args.embD_emb_models_dir)
    elif model_name == 'editembD': 
        model = EditembD(model_args.alpha, model_args.editembD_emb_models_dir)
    elif model_name == 'tutorSS':
        model = TutorSS()
    elif model_name == 'neurSS':
        model = NeurSS(model_args.synth_run, model_args.lr, 
            model_args.epochs, model_args.freeze, 
            model_args.device, model_args.batch_size, model_args.r, model_args.neurSS_emb_models_dir)
    elif model_name == 'symSS':
        model = SymSS()
    else : 
        print('incorrect model name')
        exit(1)

    return model 


def evaluate_instance(instance, model):
    _, code_A, task_B, options = instance 
    pred = model.score(code_A, task_B, options)
    return pred


def evaluate(test_instances, labels, model_name, model_args):
    t1 = time.time()
    model = select_model(model_name, model_args)
    task_A = test_instances[0][0]
    model.observe_task(task_A)    
    if model_name == 'tutorSS':
        accuracy, predictions = model.get_accuracy(
            tutor_id = model_args.tutor_id)
        return accuracy, predictions, model  
    else : 
        predictions = []
        for instance in test_instances:

            _, code_A, task_B, options = instance 
            model.observe_stu(code_A)
            if model_name == 'neurSS':
                pred = model.score(code_A, task_B, options)
            else : 
                pred = model.score(task_B, options)
            predictions.append(pred) 
            
        predictions = np.array(predictions)

    t2 = time.time()
    logging.info(f'Evaluation took {t2 - t1:.2f} seconds')
    accuracy = score(test_instances, labels, predictions)
    return accuracy, predictions, model



def load_benchmark(benchmark_file, source_task, benchmark_samples):

    if benchmark_file is not None : 
        save_folder = OUTPUT_DIR + f'/saved_benchmarks/{source_task}/{benchmark_file}/'

        if os.path.exists(save_folder)    :

            path = save_folder + 'eval_instances.p'

            with open(path, 'rb') as f:
                eval_instances = pickle.load(f)

            with open(save_folder + 'instance_info.tsv', 'r') as f : 
                instance_info = pd.read_csv(f, sep = '\t')

            with open(save_folder + 'codes.tsv', 'r') as f : 
                codes = pd.read_csv(f, sep = '\t')


        else : 
       
            benchmark = Benchmark(source_task, benchmark_samples)
       
            os.makedirs(save_folder, exist_ok=True)

            benchmark.tasks.to_csv(save_folder + '/tasks.tsv', sep = '\t', index = False)            
            benchmark.codes.to_csv(save_folder + '/codes.tsv', sep = '\t', index = False)
            benchmark.instance_info.to_csv(save_folder + '/instance_info.tsv', 
                sep = '\t', index = False)
        
            with open(save_folder + '/eval_instances.p', 'wb+') as f:
                pickle.dump(benchmark.eval_instances, f)

            eval_instances = benchmark.eval_instances
            instance_info = benchmark.instance_info
            codes = benchmark.codes

    if benchmark_file is None : 
        benchmark = Benchmark(source_task, benchmark_samples)
        eval_instances = benchmark.eval_instances
        instance_info = benchmark.instance_info
        codes = benchmark.codes

    test_instances, labels = eval_instances

    return test_instances, labels, instance_info, codes


def objective(trial, train_instances, train_labels, model_name, model_args):
    if model_name == 'editembD': 
        args_trial = add_editembD_trial_args(trial)
    elif model_name == 'neurSS' : 
        args_trial = add_neurSS_trial_args(trial)
    else :
        args_trial = dict()

    args_train = update_arguments(destination = model_args, source = args_trial)
    accuracy, _, _ = evaluate(train_instances, train_labels, model_name, args_train)
    return accuracy 

def optimize_over_train(train_instances, train_labels, num_trials, model_name, model_args):

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        sampler = sampler, 
        direction = 'maximize'
    )
    concrete_objective = lambda trial : objective(trial, train_instances, train_labels, model_name, model_args)
    study.optimize(concrete_objective,   n_trials = num_trials)
    best_parameters = study.best_trial.params        
    return best_parameters, study

def update_arguments(destination, source):

    destination_copy = deepcopy(destination)
    dargs = vars(destination_copy)
    if not isinstance(source, dict):
        source = vars(source)
    dargs.update(source)
    dargs = SimpleNamespace(**dargs)

    return destination_copy 



def save_run(args, path, accuracy, predictions, labels, model_args, study = None):
    os.makedirs(path, exist_ok = True)
    with open(path + 'commandline_args.json', 'w+') as f:
        json.dump(args.__dict__, f, indent = 2)

    with open(path + 'results.json', 'w+') as f:
        json.dump({
            "accuracy" : accuracy, 
            "model_name" : args.model_name,
            "model_args" : model_args.__dict__,
            "source_task" : args.source_task,
            "benchmark_samples" : args.benchmark_samples,
            "benchmark_file" : args.benchmark_file,
            "predictions" : predictions.tolist(),
            "labels"       : labels.tolist() 

        }, f , indent = 2)
    if study is not None : 
        with open(path + 'study.p', 'wb+') as f:
            pickle.dump(study, f)
    

def save_run_cross_validation(args, path, mean_train_accuracy, mean_test_accuracy, gathered_fold_info, 
    accuracy_train, accuracy_test):
    os.makedirs(path, exist_ok = True)

    with open(path + 'commandline_args.json', 'w+') as f:
        json.dump(args.__dict__, f, indent = 2)

    with open(path + f'results_per_fold.json', 'w+') as f: 
        json.dump(gathered_fold_info, f, indent = 2)

    with open(path + 'results.json', 'w+') as f:
        json.dump({
            "mean_train_accuracy" : mean_train_accuracy, 
            "mean_test_accuracy" : mean_test_accuracy, 
            "model_name" : args.model_name,
            "source_task" : args.source_task,
            "benchmark_samples" : args.benchmark_samples,
            "benchmark_file" : args.benchmark_file,
            "accuracy_train" : accuracy_train, 
            "accuracy_test" : accuracy_test, 
        }, f , indent = 2)
