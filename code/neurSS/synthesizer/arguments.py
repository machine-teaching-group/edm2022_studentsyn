
import argparse 
import os 
import json 
import numpy as np 
import random 
import torch 
from code.utils.utils import *
from .data import load_generated_data_info

def parse_arguments():

    parser = argparse.ArgumentParser()

    add_model_arguments(parser)
    add_beam_search_arguments(parser)
    add_data_arguments(parser)
    add_training_arguments(parser)

    args = parse(parser)

    return args

def add_model_arguments(parser):

    parser.add_argument("--kernel_size", type=int, 
            default=3, help = "CNN kernel size")

    parser.add_argument("--conv_stack", type=str, 
            default="64,64,64", help = 'number of Channels for stack of ResBlocks,\
            for different sizes e.g. 64,32, an extra CNN is added for conversion')

    parser.add_argument("--fc_stack", 
            type=str, default="512", 
            help = "CNN dim -> fc_stack, apply linears after grid CNN encoder, fc_stack[-1] = emb dim")
 
    parser.add_argument("--tgt_embedding_size", 
            type=int,default=256, 
            help = "token embedding dim")
 
    parser.add_argument("--lstm_hidden_size", type=int,
            default=256,
            help="Dimension of the LSTM hidden units. "
            "Default: %(default)s")
 
    parser.add_argument("--nb_lstm_layers", type=int,
            default=2,
            help="Nb of LSTM layers. "
            "Default: %(default)s")
 
    parser.add_argument("--syntax", type=str,
            default=None,
            help="syntax checking mode", choices = ['checker','learned'])

def add_beam_search_arguments(parser):

    parser.add_argument("--top_k", type = int, help="Nb of top predictions in beam search", default = 1)
    parser.add_argument("--beam_size",type =int,help="beam search width (frontier)", default = 1)

def add_data_arguments(parser):

    parser.add_argument("--data", type = str, help="dataset name", required = True)
    parser.add_argument("--rotate",type = bool , default = True, help = 'use all 4 rotations of a grid as input')
    parser.add_argument("--num_io_eval",type = int , default = 1, help = 'number of parallel grids used in evaluation')
    parser.add_argument("--num_io_train",type = int , default = 1, help = 'number of parallel grids used in training')
    parser.add_argument("--override",action = 'store_const', default = False, const = True, help ='ignore cached data, read input again')
    parser.add_argument("--num_samples",type = int , default = None, help = 'limit on number of training points to read from dataset')
    parser.add_argument("--task",type = str , default = 'hoc4')
    
def add_training_arguments(parser):
    parser.add_argument("--beta_syntax",type = float , default = 1e-5)
    parser.add_argument("--batch_size",type = int , default = 64)
    parser.add_argument("--epochs",type = int , default = 10)
    parser.add_argument("--lr",type = float , default = 1e-4)
    parser.add_argument("--val_freq",type = int , default = 10)
    parser.add_argument("--run_id",type = str , default = 'test', help = 'run id for saving in results folder')
    parser.add_argument("--device" ,default = 'cuda', type = str)
    parser.add_argument("--wd", type = float, default = 0)
    parser.add_argument("--val_metric" ,default = 'sem', type = str, choices = ['exact', 'sem', 'gen', 'min_sem'])

def parse(parser, random_seed = None):

    if random_seed is not None: 
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)

    args = parser.parse_args()
    args.conv_stack = [int(x) for x in args.conv_stack.split(',')]
    args.fc_stack = [int(x) for x in args.fc_stack.split(',')]

    if args.run_id is None : args.run_id = 'test'

    save_path = NEURSS_DIR + f'/{args.task}/runs/{args.run_id}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    log_file = save_path + '/logs.txt'        
    set_logger(str(log_file))

    args.data = NEURSS_DIR + f'/{args.task}/training_data/{args.data}/'
    args.train_data = args.data + 'train.data'
    args.val_data =  args.data + 'val.data'
    args.test_data =  args.data + 'test.data'
      
    with open(save_path + '/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    data_info = load_generated_data_info(args.data + '/info.json')

    model_args = {

        "kernel_size" : args.kernel_size, 
        "conv_stack" : args.conv_stack, 
        "fc_stack" : args.fc_stack,
        "tgt_vocabulary_size" : data_info["vocab_size"],
        "tgt_embedding_dim": args.tgt_embedding_size,
        "lstm_hidden_size": args.lstm_hidden_size,
        "nb_lstm_layers": args.nb_lstm_layers,
        "check_syntax_mode" :  args.syntax,
        "vocab" : data_info["vocab"],
        "IMG_SIZE" : data_info["IMG_SIZE"]
    }

    with open(save_path + '/model_args.json', 'w') as f:
        json.dump(model_args, f, indent=2)

    with open(save_path + '/data_info.json', 'w') as f:
        json.dump(data_info, f, indent=2)
    args.save_path = save_path
    return args, model_args, data_info