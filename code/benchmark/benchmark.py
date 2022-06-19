import random
import numpy as np 
import pandas as pd 
import glob 
import string 
from random import shuffle 
from copy import deepcopy 
from code.utils.utils import * 
import json 




class Benchmark:
    
    def __init__(self, source_task, number_benchmark_instances):
        benchmark_task_dir = BENCHMARK_DIR + f'/{source_task}/'

        behaviours = pd.read_csv(benchmark_task_dir + 'behaviours.tsv', sep='\t+', engine = 'python')   
        self.behaviours = dict(zip(behaviours.misconceptions, behaviours.explanations))

        benchmark_codes_df = pd.DataFrame([])


        solution_df, task_names, task_paths, solution_paths = read_task_sol(benchmark_task_dir)
        students_df = read_stu(benchmark_task_dir)
        rand_df = read_rand(benchmark_task_dir)

        benchmark_codes_df = pd.concat([benchmark_codes_df, students_df], ignore_index = True)
        benchmark_codes_df = pd.concat([benchmark_codes_df, solution_df], ignore_index = True)
        benchmark_codes_df = pd.concat([benchmark_codes_df, rand_df], ignore_index = True)

        get_student_name_fn = lambda x : x.split('/')[-1].split('_')[-1].split('.')[0]
        benchmark_codes_df['stu_id'] = benchmark_codes_df['code_path'].apply(get_student_name_fn)
        
        number_all_codes = len(benchmark_codes_df)
        number_of_stds = benchmark_codes_df['stu_id'].str.contains('stu').sum()
        number_of_rands = benchmark_codes_df['stu_id'].str.contains('rand').sum()
        number_of_sols = benchmark_codes_df['stu_id'].str.contains('solution').sum()

        print(f'Found {number_all_codes} codes in benchmarks:')
        print(f'\t{number_of_stds} students')
        print(f'\t{number_of_rands} random')
        print(f'\t{number_of_sols} solutions')

        benchmark_codes_idxs_df = benchmark_codes_df.copy()
        benchmark_codes_idxs_df['code_idx'] = list(range(number_all_codes))
        instance_info_df = sample_benchmark_instances(benchmark_codes_idxs_df, number_benchmark_instances, task_names, source_task)

        tasks = []
        for tpath in task_paths : 
            with open(tpath, 'r') as f : 
                task = f.read()
            tasks.append(task)

        codes = []
        code_paths = benchmark_codes_idxs_df['code_path'].tolist()
        for cpath in code_paths : 
            with open(cpath, 'r') as f : 
                code = json.load(f)
            codes.append(code)

        solutions = []
        for spath in solution_paths : 
            with open(spath, 'r') as f : 
                sol_code = json.load(f)
            solutions.append(sol_code)

        benchmark_codes_idxs_df['code_path'] = codes
        benchmark_codes_idxs_df = benchmark_codes_idxs_df.rename(columns = {"code_path" : "code"})    

        task_df = pd.DataFrame(data = {"name" : task_names, "task": tasks, "solution" : solutions})

        self.codes = benchmark_codes_idxs_df
        self.tasks = task_df
        self.instance_info = instance_info_df

        self.eval_instances = create_eval_instances(self.instance_info, self.codes, self.tasks)

def create_eval_instances(instances, codes, tasks):

    instances_info = instances.values 
    labels = instances['label'].values
    test_instances = []
    for inst in instances_info: 

        options = codes.iloc[ inst[3:-2] ].code.values
        code_A = codes.iloc[ inst[2] ].code
        task_A = tasks[ tasks['name'] == inst[0] ].iloc[0].to_dict()
        task_B = tasks[ tasks['name'] == inst[1] ].iloc[0].to_dict()
        test_instances.append([task_A, code_A, task_B, options])

    test_instances = np.array(test_instances, dtype = object)
    return test_instances, labels 


def read_stu(benchmark_data_path):

    benchmark_students_df = pd.read_csv(benchmark_data_path + 'info.tsv', sep='\t+', engine = 'python')
    benchmark_students_df['stu'] = benchmark_data_path + '/stu/' + benchmark_students_df['stu'] + '.json'
    benchmark_students_df = benchmark_students_df.rename(columns = {"stu" : "code_path"})    
    benchmark_students_df['notes'] = 'stu'

    return benchmark_students_df


def read_task_sol(benchmark_data_path):

    solution_paths = glob.glob(benchmark_data_path + 'solution/*')
    task_paths = glob.glob(benchmark_data_path + '/task/*')
    solution_paths.sort()
    task_paths.sort()

    extract_task_fn = lambda path : path.split('/')[-1].split('.')[0].split('_')[0]

    solution_tasks = [extract_task_fn(sol_path) for sol_path in solution_paths]
    solution_notes = ["sol" for _ in range(len(solution_paths)) ] 

    task_names = [extract_task_fn(path) for path in task_paths]

    solution_df = pd.DataFrame(
        {
            "code_path" : solution_paths, 
            "task" : solution_tasks,
            "notes" : solution_notes,                
        }
    )
    return solution_df, task_names, task_paths, solution_paths 


def read_rand(benchmark_data_path):

    rand_info_df = pd.read_csv(benchmark_data_path + '/rand_info.tsv', sep = '\t')  
    random_ids = rand_info_df['rand_id'].tolist()
    random_task_names = rand_info_df['task'].tolist()
    random_notes = rand_info_df['notes'].tolist()

    random_std_paths = [benchmark_data_path + f'/rand/{rand_task}_rand{rand_id}.json'
        for rand_task, rand_id in zip(random_task_names, random_ids)]

    rand_df =  pd.DataFrame(
        {
            "code_path" : random_std_paths, 
            "task" : random_task_names,
            "notes" : random_notes
        }
    )

    return rand_df 




def sample_benchmark_instances(benchmark_students_df, number_benchmark_instances, task_names, source_task):

    misconceptions = benchmark_students_df['mis'].unique()
    misconceptions = list(filter(lambda misc_id : not np.isnan(misc_id), misconceptions))

    print(f'Sampling {number_benchmark_instances} unique benchmark instances')
    benchmark_instances = []
    for _ in range(number_benchmark_instances):
        instance = create_benchmark_instance_df(source_task, benchmark_students_df, task_names, misconceptions)
        benchmark_instances.append( instance )

    full_benchmark_values = np.concatenate(benchmark_instances, axis = 0)
    full_benchmark_df = pd.DataFrame(full_benchmark_values, columns = instance.columns)
    print(f'Created benchmark with {len(full_benchmark_df)} test samples')

    return full_benchmark_df


def create_benchmark_instance_df(source_task, benchmark_students_df, task_names, misconceptions):

    lowercase_alphabet = string.ascii_lowercase[:10]

    students_B =  [f'code_B_{n}' for n in lowercase_alphabet]
    columns  = ['task_A','task_B','code_A'] + students_B + ['label', 'mis']

    data = []

    source_task_df = benchmark_students_df[benchmark_students_df['task'] == source_task]        
    task_names = task_names.copy()
    task_names.remove(source_task)

    for target_task in task_names : 

        task_idxs = benchmark_students_df['task'] == target_task
        target_task_df = benchmark_students_df[task_idxs]        
        for misc_current_id in misconceptions : 

            source_misconception_df = source_task_df[source_task_df['mis'] == misc_current_id]
            target_misconception_df = target_task_df[target_task_df['mis'] == misc_current_id]

            std_A_idx = source_misconception_df.sample(1)['code_idx'].item()
            stu_id_A = source_misconception_df[source_misconception_df['code_idx'] == std_A_idx]['stu_id'].item()
            std_B_target_idx =  target_misconception_df[target_misconception_df['stu_id'] == stu_id_A].sample(1)['code_idx'].item()

            std_rest = []
            misconceptions_rest = deepcopy(misconceptions)
            misconceptions_rest.remove(misc_current_id)
            for misc_rest_id in misconceptions_rest:
                target_misconception_df_other = target_task_df[target_task_df['mis'] == misc_rest_id]

                other_std_idx = target_misconception_df_other.sample(1)['code_idx'].item()  
                std_rest.append(other_std_idx)

            rand_info_df_target = benchmark_students_df[
                (benchmark_students_df['task'] == target_task) & 
                (benchmark_students_df['stu_id'].str.contains('rand'))]

            if source_task == 'hoc4' : 

                stds_rand = [ rand_info_df_target[rand_info_df_target['notes'] == f'edit_{i}'].sample(1)['code_idx'].item()
                    for i in [2, 5, 10] ]
            else : 

                stds_rand_diff_structure = rand_info_df_target[
                    (rand_info_df_target['notes'] == 'only_if') |
                    (rand_info_df_target['notes'] == 'only_while') |
                    (rand_info_df_target['notes'] == 'basic_actions')
                ]

                rand_diff_structure_codes = stds_rand_diff_structure['code_idx'].tolist()
                stds_rand = random.sample(rand_diff_structure_codes, 2)
                stds_rand +=  [rand_info_df_target[rand_info_df_target['notes'] == f'edit_2'].sample(1)['code_idx'].item()]

            solution = target_task_df[target_task_df['notes'] == 'sol']['code_idx'].item()
            all_row_codes = stds_rand + [solution] + [std_B_target_idx] + std_rest 
            shuffle(all_row_codes)

            label = all_row_codes.index(std_B_target_idx)

            data.append( 
                [source_task, target_task] + [std_A_idx] + all_row_codes + [label, misc_current_id]
            )
    
    benchmark_instance_df = pd.DataFrame(data, columns = columns)

    return benchmark_instance_df 


if __name__ == '__main__' : 
    x = Benchmark('hoc4',2)
    import pdb ; pdb.set_trace()