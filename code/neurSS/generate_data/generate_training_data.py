import argparse
import os
import json
import numpy as np 
from tqdm import tqdm

from .code_generator import CodeGenerator
from .generate_data_hoc4 import generate_grids_hoc4
from .generate_data_hoc18 import generate_grids_hoc18
from code.utils.parser.world import World
from code.utils.utils import NEURSS_DIR, set_logger

import os 
import logging 
from multiprocessing import Pool 

def get_args():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--num_examples', type = str, default = '5,1,1', help = 'how many complete examples to generate')
    arg_parser.add_argument('--num_io', type=int, default=2, help='number of io grids for every example')
    arg_parser.add_argument('--task', type = str, default = 'hoc4')
    arg_parser.add_argument('--dataset_name', type=str, help = 'data storing directory', required = True)
    arg_parser.add_argument('--grid_size', type=int, default=12)
    

    arg_parser.add_argument('--num_iterations', type=int, default=1000, help='number of random rollouts to execute')
    arg_parser.add_argument("--patience",type = int, default = 100, help = 'how many consecutive until we give up on grid generation for code')

    arg_parser.add_argument("--init",type = str, default = 'full')
    arg_parser.add_argument("--num_cores",type = int, default = 1)
    arg_parser.add_argument("--debug", const = True, action='store_const')

    args = arg_parser.parse_args()

    return args 

def main():
   

    args = get_args()
    data_path = NEURSS_DIR + f'/{args.task}/training_data/{args.dataset_name}/'
    args.data_path = data_path 
    os.makedirs(data_path, exist_ok=True)
    set_logger(data_path + 'log.txt')
    
    num_train, num_val, num_test = map(int, args.num_examples.split(','))
    for data_name, num_data_examples in zip(['train', 'val', 'test'],[num_train, num_val, num_test]):

        logging.info(f'Init parallel generation with {args.num_cores} cores')
        logging.info(f'Generating {num_data_examples} codes')
        logging.info(f'with diverse tasks : {args.num_io}')

        with open(args.data_path + '/' + 'args' + ".txt",'w+') as f :
            json.dump(args.__dict__, f, indent = 2)


        code_generator = CodeGenerator()

        if args.task == 'hoc18':
            tokens_raw = code_generator.token_names.__dict__.values()
        else : 
            dico = code_generator.token_names.__dict__
            tokens_raw = [
                dico["t_M_LBRACE"], dico["t_M_RBRACE"],
                dico["t_RUN"], dico["t_DEF"], 
                dico["t_MOVE"], dico["t_TURN_RIGHT"],
                dico["t_TURN_LEFT"]]

        tokens =  [t.replace('\\','') for t in tokens_raw]

        with open(args.data_path + '/info.json','w+') as f : 
            json.dump(
                {
                "tokens" : tokens,
                "grid_size" : args.grid_size,
                "img_feat" :  args.grid_size * args.grid_size * World.VECTOR_SIZE,
                "img_size" : (World.VECTOR_SIZE,args.grid_size, args.grid_size)
                }, f, indent =2) 


        if args.num_cores > 1 : 

            results = []
            pool = Pool(args.num_cores)
            div = num_data_examples // args.num_cores
            rem = num_data_examples % args.num_cores 
            samples_per_core = [div]*(args.num_cores)
            samples_per_core[0] = samples_per_core[0] + rem 

            for worker_idx in range(1, args.num_cores + 1):
                seed = 42 + worker_idx
                wait_res = pool.apply_async(generate_data, [samples_per_core[worker_idx-1], code_generator,  args,
                seed])
                results.append(wait_res)        

            data_samples = []
            for wait_res in results : 
                data_samples += wait_res.get()
        else : 
            data_samples = generate_data(num_data_examples, code_generator,  args, seed = 42)

        data_file = args.data_path + '/' + data_name + ".data"    
        with open(data_file,'w+') as f :
            for sample in data_samples: 
                f.write(json.dumps(sample) + '\n')

def generate_data(num_examples, code_generator,  args, seed):

    random_generator = np.random.RandomState(seed)
    data_samples = []
    num_generated = 0


    with tqdm(total=num_examples) as pbar:

        while num_generated <  num_examples :

            candidate_code = code_generator.random_code(random_generator, task = args.task)
            candidate_code_str = ' '.join(candidate_code)

            if args.task == 'hoc18':
                examples = generate_grids_hoc18(candidate_code_str, args.grid_size, random_generator,  args.num_io, 
                args.patience, args.num_iterations, debug = args.debug)
            
            elif args.task == 'hoc4' :  
                examples = generate_grids_hoc4(
                    candidate_code_str, random_generator, args.num_io, args.grid_size, 
                    args.patience, debug = args.debug)                   

            if examples == [] : continue 
            assert len(examples) == args.num_io

            data_samples.append( 
                {
                    "program_tokens" : candidate_code, 
                    "size" : len(candidate_code),
                    "examples" : examples
                })

            num_generated += 1 
            pbar.update(1)

    assert len(data_samples) == num_examples
    return data_samples     



if __name__ == '__main__':
    main()
