from code.utils.utils import OUTPUT_DIR
import json 
import argparse 
import numpy as np 
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def significane_analysis(args):
    load_path = OUTPUT_DIR + f'/discriminative/evaluation_{args.benchmark_file}_{args.source_task}/final_results.json'
    with open(load_path, 'r') as f :
        results = json.load(f)

    for model, acc in results.items():
        assert len(acc['accuracies']) >= 2

    # calculate significance of results 
    F, p = stats.f_oneway(
        results['editD']['accuracies'], 
        results['editembD']['accuracies'],
        results['neurSS']['accuracies'],
        results['symSS']['accuracies']
    )

    print('F statistic value : ', F)
    print('p value : ', p)
    accuracy_models = []
    accuracy_values = []
    for model in ['editD','editembD','neurSS','symSS']:
        acc = results[model]['accuracies']
        num_values = len(acc)
        accuracy_values += acc
        accuracy_models += [model]*(num_values)
 
    tukey = pairwise_tukeyhsd(accuracy_values, accuracy_models, alpha=0.05)
    print(tukey)
    
    for model in ['randD', 'editD', 'editembD', 'neurSS', 'symSS', 'tutorSS']:
        acc = np.array(results[model]['accuracies'])*100
        scipy_stderr = stats.sem(acc)
        print(f'{model}:  {np.mean(acc)} +-',scipy_stderr) 
               

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_file', default = 'RESULTS', help = 'file used for the evaluation')
    parser.add_argument('--source_task', default = 'hoc4', help = 'task used for the evaluation')
    
    args = parser.parse_args()
    significane_analysis(args)
