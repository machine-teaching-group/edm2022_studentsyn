import argparse
import json 
from code.exp_discriminative.train_utils import select_model
from code.utils.utils import OUTPUT_DIR, set_logger
from code.benchmark.benchmark import Benchmark 
from types import SimpleNamespace
from code.baselines.algo_tutorSS import add_tutorSS_parameters
from code.neurSS.algo_neurSS import add_neurSS_parameters
import os 
import time 
import logging

def load_model(model_name, run_path):
    with open(run_path + 'results.json', 'r') as f:
        model_args = json.load(f)['model_args']
    model_args = SimpleNamespace(**model_args)
    model = select_model(model_name, model_args)
    model.load(run_path)

    return model 


def generate_main(args, model_args_dict):

    path = OUTPUT_DIR + f'generative/{args.save_name}/'
    os.makedirs(path, exist_ok=True)
    set_logger(path + 'log.txt')


    model_args = model_args_dict.get(args.model_name, [])
    args_dict = vars(args)
    model_args = argparse.Namespace(**{key : args_dict[key] for key in model_args })    
    run_path = OUTPUT_DIR + f'/discriminative/{args.load_run}/'


    if args.model_name == 'neurSS' : 
        if os.path.exists(run_path) :
            model = load_model(args.model_name, run_path)
        else : 

            logging.info(f'neurSS model path {run_path} does not exist.')
            raise Exception('path does not exist, specify with `--load_run`')
    else : 
        model = select_model(args.model_name, model_args)

    
    benchmark = Benchmark(args.source_task, 1)

    tasks = benchmark.tasks
    source_task = tasks[(tasks.name == args.source_task)].iloc[0].to_dict()
    codes = benchmark.codes 
    students = codes[codes.stu_id.str.contains('stu')]
    students_source = students[students.task == source_task["name"]]
    target_tasks = tasks

    model.observe_task(source_task)

  

    for student in students_source.itertuples() :
        codes_stu = students[students.stu_id == student.stu_id]
        code_A = codes_stu[codes_stu.task == source_task["name"]].code.item()
        results_stu = []

        logging.info(f'Generating for student {student.stu_id}')

        for target_task in target_tasks.itertuples():
            t1 = time.time()

            target_task = target_task._asdict()
            logging.info(f'\tfor target {target_task["name"]}')

            code_B_label = codes_stu[codes_stu.task == target_task["name"]].code.item()

            if args.model_name == 'tutorSS':

                gen_codes = model.generate(target_task["name"], student.stu_id, 
                tutor_id = args.tutor_id)
                probs = None

            elif args.model_name == 'neurSS':

                gen_codes, probs = model.generate(code_A, target_task, beam_size = args.beam_size, top_k = args.top_k)    

            else : 

                model.observe_stu(code_A)
                gen_codes, probs = model.generate(target_task, top_k = args.top_k)    

            t2 = time.time()
            logging.info(f'generation took {t2 - t1:.2f} seconds.')

            mis = int(student.mis)


            gen_info = {
                "model_name" : args.model_name,
                "source" : source_task["name"],
                "target" : target_task["name"],
                "mis"    : mis,
                "behavior" : benchmark.behaviours[f'mis_{mis}'],
                "norm_prob" : probs,
                "code_A" : code_A,
                "code_B_gen" : gen_codes,
                "code_B_label" : code_B_label
            }
            results_stu.append(gen_info)
  
            os.makedirs(path + student.stu_id, exist_ok=True)

            for gen_info in results_stu:

                with open(path + f'{student.stu_id}/{gen_info["target"]}.json', 'w+') as f:
                    json.dump(gen_info, f, indent = 2)

            if args.model_name == 'symSS':
                model.save_stu_grammars(path + student.stu_id + '/')
            

    with open(path + 'args.json','w') as f:
        json.dump(args.__dict__, f, indent = 2)

def add_args_generate(parser):
    parser.add_argument("--save_name", default = 'results_generate', help = 'save results in /outputs/generative/{save_name}/')
    parser.add_argument("--source_task", default = 'hoc4', choices = ['hoc4', 'hoc18'])
    parser.add_argument("--model_name", required = True, type = str, help ='model to use for generation', choices = ['neurSS','symSS','tutorSS'])
    parser.add_argument('--top_k', type = int, default = 3,  help = 'number of top-scoring codes to generate')
    parser.add_argument('--beam_size', type = int, default = 64, help = 'beam size used for generation with beam search sampling for NeurSS')
    parser.add_argument('--tutor_id', type = str, default = None, help = 'specify tutor id for generation',
        choices = ['TutorSS1','TutorSS2','TutorSS3'])

    parser.add_argument("--load_run", default = None, type = str,
        help = 'run name in outputs/discriminative/ to load pretrained neurSS student models, required for neurSS generation')


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    model_args_dict = {}
    add_args_generate(parser)
    args = parser.parse_args()    
    generate_main(args, model_args_dict)

