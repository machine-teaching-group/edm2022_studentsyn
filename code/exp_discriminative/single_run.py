import argparse 
from code.exp_discriminative.train_utils import evaluate, load_benchmark, save_run, optimize_over_train, update_arguments, save_run_cross_validation
from code.baselines.algo_editembD import add_editembD_parameters
from code.baselines.algo_embD import add_embD_parameters
from code.neurSS.algo_neurSS import add_neurSS_parameters
from code.baselines.algo_tutorSS import add_tutorSS_parameters

from code.utils.utils import OUTPUT_DIR, set_logger

from sklearn.model_selection import KFold 
import numpy as np 
import logging 
import os 
import time

def run_cross_validation(benchmark_file, source_task, benchmark_samples, num_folds, num_trials, model_name, model_args, optimize):
    t1 = time.time()
    instances, labels, _, _ = load_benchmark(benchmark_file, source_task, benchmark_samples)
    number_instances = len(instances)
    train_indexes, test_indexes = split_into_folds(number_instances, num_folds)
    accuracy_train = []
    accuracy_test = []
    gathered_fold_info = {}

    for fold_id in range(num_folds):
        logging.info('#'*50)
        logging.info(f'Fold #{fold_id+1}')
        logging.info('#'*50)
        train_idxs, test_idxs = train_indexes[fold_id], test_indexes[fold_id]

        train_instances, train_labels = instances[train_idxs], labels[train_idxs]
        test_instances, test_labels = instances[test_idxs], labels[test_idxs]

        if optimize : 
            best_parameters, _ = optimize_over_train(train_instances, train_labels, num_trials, model_name, model_args)
            opt_args = update_arguments(destination = model_args, source = best_parameters)
        else : 
            opt_args = model_args
            best_parameters = model_args.__dict__

        acc_train, preds_train, _ = evaluate(train_instances, train_labels, model_name, opt_args)
        acc_test, preds_test, _ = evaluate(test_instances, test_labels, model_name, opt_args)
        accuracy_train.append(acc_train)        
        accuracy_test.append(acc_test)        

        fold_info = {
            "accuracy_train": acc_train,
            "accuracy_test": acc_test, 
            "best_parameters"   : best_parameters,
            "preds_train" : preds_train.tolist(), 
            "preds_test" : preds_test.tolist(),
            "labels_train" : train_labels.tolist(),
            "labels_test" : test_labels.tolist(),
            "train_idxs": train_idxs.tolist(),
            "test_idxs": test_idxs.tolist()
        }
        gathered_fold_info[f'fold_{fold_id+1}'] = fold_info
        logging.info(f'Train acc : {acc_train}')
        logging.info(f'Test acc : {acc_test}')

    mean_train = np.mean(accuracy_train)
    mean_test = np.mean(accuracy_test)
    
    logging.info(f"Mean train accuracy : {mean_train}")
    logging.info(f"Mean test accuracy : {mean_test}")
    t2 = time.time()
    logging.info(f'Cross-validation procedure took {t2 - t1} seconds')
    return mean_train, mean_test, gathered_fold_info, accuracy_train, accuracy_test

def split_into_folds(number_instances, num_folds):

    kfold = KFold(num_folds)
    indexes = np.arange(number_instances)
    train_indexes = []
    test_indexes = []
    for test_idxs, train_idxs in kfold.split(indexes):
        test_indexes.append(test_idxs)
        train_indexes.append(train_idxs)

    return train_indexes, test_indexes


def run_single(benchmark_file, source_task, benchmark_samples, optimize, num_trials, model_name, model_args):

    instances, labels, instance_info, codes = load_benchmark(benchmark_file, source_task, benchmark_samples)
    if optimize :
        best_parameters, study = optimize_over_train(instances, labels, num_trials, model_name, model_args)       
        opt_model_args = update_arguments(destination = model_args, source = best_parameters)
    else : 
        study = None
        opt_model_args = model_args


    accuracy, predictions, model = evaluate(instances, labels, model_name, opt_model_args)


    logging.info(f'Mean accuracy: {accuracy}')

    return accuracy, predictions, labels, study, model, opt_model_args

def add_training_arguments(parser):
    parser.add_argument("--source_task", default = 'hoc4', choices = ['hoc4','hoc18'])
    parser.add_argument("--benchmark_file", default = None, help = '''
        save/load a set of generated benchmark instances from /outputs/saved_benchmarks/{benchmark_file}''')
    parser.add_argument("--benchmark_samples", default = 2, type = int, help = '''
    number of full benchmark instances to sample, the final benchmark will contain {benchmark_samples} * 18 instances''')
    parser.add_argument("--model_name", required = True, choices = ['randD','editD', 'embD','editembD','neurSS','symSS', 'tutorSS'])
    parser.add_argument("--run_name", default = 'results_single_run', help = 'save results in /outputs/discriminative/{run_name}')


    parser.add_argument("--optimize", default = False, action = "store_const", const = True, 
        help = 'find optimal model parameters')

    parser.add_argument("--cross_validate", default = False, action = "store_const", const = True, 
        help = 'use a cross-validation procedure')

    parser.add_argument("--num_folds", default = 10, type = int, help = 'number of folds used for cross validation')
    parser.add_argument("--num_trials", default = 2, type = int, help = 'number of trials in optimization for parameter selection')

    parser.add_argument("--save_stud_models", default = False, action = "store_const", const = True, 
        help = 'save pre-trained student models for neurSS, to use later for generation')
 


def single_run_main(args, model_args_dict):
    args_dict = vars(args)
    model_args = model_args_dict.get(args.model_name, [])
    model_args = argparse.Namespace(**{key : args_dict[key] for key in model_args })

    save_path = OUTPUT_DIR + f'discriminative/{args.run_name}/'
 
    os.makedirs(save_path, exist_ok=True)
    set_logger(save_path + 'log.txt')

    if args.cross_validate : 
        mean_train, mean_test, gathered_fold_info, accuracy_train, accuracy_test = run_cross_validation(
            args.benchmark_file, args.source_task, args.benchmark_samples, args.num_folds, args.num_trials, 
            args.model_name, model_args, args.optimize
        )
        save_run_cross_validation(args, save_path, mean_train, mean_test, 
        gathered_fold_info, accuracy_train, accuracy_test)
        ret_acc = mean_test
    else : 
        
        accuracy, predictions, labels, study, model, opt_model_args = run_single(
            args.benchmark_file, args.source_task, args.benchmark_samples, 
            args.optimize, args.num_trials, args.model_name, model_args)
        ret_acc = accuracy

        save_run(args, save_path, accuracy, predictions, labels, opt_model_args, study)
        if args.save_stud_models: 
            model.save(save_path)

    return ret_acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_training_arguments(parser)
    model_args_dict = {}
    add_embD_parameters(parser, model_args_dict)    
    add_editembD_parameters(parser, model_args_dict)    
    add_neurSS_parameters(parser, model_args_dict)    
    add_tutorSS_parameters(parser, model_args_dict)    
    args = parser.parse_args()
    single_run_main(args, model_args_dict)
