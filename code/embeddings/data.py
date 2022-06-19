import torch 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import os 
import glob 
import json 
import pickle 
import random 
import numpy as np 

from .utils import ast_utils
from .utils import dataset_access
from .utils import graphs

from code.utils.utils import STU_DIR 

import functools
import multiprocessing

get_embed_tensor = {
    "hoc4" :    torch.tensor([0, 1, 2, 4, 7, 8]),
    "hoc18" :   torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
}
get_vocab_size = {
    "hoc4"  : 6, 
    "hoc18" : 12
    }

HIDDEN_DIM = 20


class StudentData(Dataset):

    def __init__(self, task = 'hoc4', override = False):

        task_dir = STU_DIR + task       
        ast_files = glob.glob(task_dir + '/asts/*.json')
        asts_dict = {}
        ast_idxs = []
        ast_codes = []
        for file in ast_files : 
            ast_idx = int(os.path.split(file)[1].split('.')[0])
            ast_code = json.load(open(file,'r'))
            asts_dict[ast_idx] = ast_code 
            ast_idxs.append(ast_idx)
            ast_codes.append(ast_code)

        self.ast_dict = asts_dict
        self.ast_idxs = ast_idxs
        self.ast_codes = ast_codes
        self.length = len(self.ast_idxs)

    def __len__(self):
        return self.length 

    def __getitem__(self,idx):
        return self.ast_idxs[idx], self.ast_codes[idx]



class SyntaxData(Dataset):

    def __init__(self, dataset_id, mode,override = False):

        # some dataset saving, because 
        # HOC18 takes a long time to generate 
        data_path = dataset_access.GENERATED_DIR + f'/train_data_{dataset_id}.p'
        if not override and os.path.isfile(data_path):
            print('loading data')
            with open(data_path,'rb') as handle : 
                goal, dataset = pickle.load(handle)
        else : 
            print('generating data')
            goal, dataset = generate_dataset(dataset_id, seed=111)
            with open(data_path,'wb') as handle : 
                pickle.dump((goal, dataset), handle, protocol = pickle.HIGHEST_PROTOCOL)


        ins1, ins2, ds12, ds21, _, _, _, _ = dataset[mode]

        # test we don't use here 
        # test_ins1, test_ins2, test_ds12, test_ds21, _, _, _, _ = dataset['test']
        
        ins1_tensor = [torch.tensor(i) for i in ins1]
        ins2_tensor = [torch.tensor(i) for i in ins2]
        ins1_final = pad_sequence(ins1_tensor, batch_first = True)
        ins2_final = pad_sequence(ins2_tensor, batch_first = True)
        lens1 = [len(l) for l in  ins1]
        lens2 = [len(l) for l in  ins2]
        ds12 = torch.tensor(ds12)
        ds21 = torch.tensor(ds21)
        # magic to do change of basis because 
        orig1 = ins1_final.shape
        orig2 = ins2_final.shape
        A = get_embed_tensor[dataset_id]
        ins1 = (ins1_final.view(-1,1) == A).int().argmax(dim=1).reshape(orig1)
        ins2 = (ins2_final.view(-1,1) == A).int().argmax(dim=1).reshape(orig2)

        # we need this later to embed new inputs
        # its the mapping between their AST node idxs 
        # and the models 

        self.embed_tensor = A
        self.ins1 = ins1 
        self.ins2 = ins2 
        self.lens1 = lens1 
        self.lens2 = lens2 
        self.ds12 = ds12 
        self.ds21 = ds21

    def __len__(self):
        return self.ins1.shape[0]

    def __getitem__(self, idx):  
        return self.ins1[idx], self.ins2[idx],\
            self.ds12[idx],self.ds21[idx]




def get_distances(g, node) :
    return graphs.distances_bfs(g, node)[0]


def generate_dataset(dataset_id, seed=None):
    print('Generating dataset')
    if seed is not None:
        np.random.seed(seed)

    goal = dataset_access.load_ast('0', dataset_id)

    g = dataset_access.load_graph(dataset_id)

    if g == None : 
        print('graph data not present, generating..')
        g = graphs.generate_graph(dataset_id)
  
    expected_n_nodes = 1000
    d_from_goal, _ = graphs.distances_bfs(graphs.transpose(g), goal)

    def compute_d_dist(ds):    
        d_counts = dict()
        for k, v in ds.items():
            if v not in d_counts:
                d_counts[v] = 0
            d_counts[v] += 1.0
        for k, v in d_counts.items():
            d_counts[k] /= len(ds)
        
        return d_counts

    d_from_goal_dist = compute_d_dist(d_from_goal)
    d_expected = 0
    for k, v in d_from_goal_dist.items():
        d_expected += k * v

    p_sampled = expected_n_nodes / len(g)
    def p_d_weighted(d):
        return 0.1 * p_sampled / d_from_goal_dist[d]

    node_ints = []
    for node_int in g.keys():
        if np.random.rand() < p_d_weighted(d_from_goal[node_int]):
            node_ints.append(node_int)

    if goal.to_int() not in node_ints:
        node_ints.append(goal.to_int())

    print('Sampled {} nodes'.format(len(node_ints)))
    print(compute_d_dist({ k: d_from_goal[k] for k in node_ints }))

    ds12 = []
    ds21 = []
    ds01 = []
    ds02 = []
    ds10 = []
    ds20 = []
    ins1 = []
    ins2 = []
    random.shuffle(node_ints)
    nodes = [ ast_utils.ASTNode.from_int(i) for i in node_ints ]
    tokens = [ np.array(n.tokenize()) for n in nodes ]

    print('Computing distances')
    print('# nodes in graph:', len(g))
    n_e = sum(len(v) for _, v in g.items())
    print('# edges in graph:', n_e)

    with multiprocessing.Pool(64) as p:
        node_distances = p.map(functools.partial(get_distances, g), nodes)

    print('DONE')

    for i, (node_i, tokens_i) in enumerate(zip(nodes, tokens)):
        print(i, '/', len(nodes))
        d_from_i = node_distances[i]
        for j, (node_j, tokens_j) in enumerate(zip(nodes, tokens)):
            d_from_j = node_distances[j]
            dij = d_from_i[node_j.to_int()]
            dji = d_from_j[node_i.to_int()]
            di0 = d_from_i[goal.to_int()]
            d0i = d_from_goal[node_i.to_int()]
            dj0 = d_from_j[goal.to_int()]
            d0j = d_from_goal[node_j.to_int()]
            ds12.append(dij)
            ds21.append(dji)
            ds01.append(d0i)
            ds02.append(d0j)
            ds10.append(di0)
            ds20.append(dj0)
            ins1.append(np.array(tokens_i))
            ins2.append(np.array(tokens_j))
 
 
 
    ds12 = np.array(ds12)
    ds21 = np.array(ds21)
    ds01 = np.array(ds01)
    ds10 = np.array(ds10)
    ds02 = np.array(ds02)
    ds20 = np.array(ds20)
    dcounts = dict()

    for v in ds12:
        if v not in dcounts:
            dcounts[v] = 0
        dcounts[v] += 1

    print('Distance counts')
    print(dcounts)

    goal = np.array(goal.tokenize())

    data = list(zip(ins1, ins2, ds12, ds21, ds01, ds10, ds02, ds20))
    random.shuffle(data)
    ins1, ins2, ds12, ds21, ds01, ds10, ds02, ds20 = zip(*data)
    ins1 = list(ins1)
    ins2 = list(ins2)
    ds12 = np.array(ds12)
    ds21 = np.array(ds21)
    ds01 = np.array(ds01)
    ds10 = np.array(ds10)
    ds02 = np.array(ds02)
    ds20 = np.array(ds20)

    N = len(ds12)
    I_train = int(0.8*N)
    I_validate = int(0.9*N)

    train_ins1 = ins1[:I_train]
    train_ins2 = ins2[:I_train]
    train_ds12 = ds12[:I_train]
    train_ds21 = ds21[:I_train]
    train_ds01 = ds01[:I_train]
    train_ds10 = ds10[:I_train]
    train_ds02 = ds02[:I_train]
    train_ds20 = ds20[:I_train]

    validate_ins1 = ins1[I_train:I_validate]
    validate_ins2 = ins2[I_train:I_validate]
    validate_ds12 = ds12[I_train:I_validate]
    validate_ds21 = ds21[I_train:I_validate]
    validate_ds01 = ds01[I_train:I_validate]
    validate_ds10 = ds10[I_train:I_validate]
    validate_ds02 = ds02[I_train:I_validate]
    validate_ds20 = ds20[I_train:I_validate]

    test_ins1 = ins1[I_validate:]
    test_ins2 = ins2[I_validate:]
    test_ds12 = ds12[I_validate:]
    test_ds21 = ds21[I_validate:]
    test_ds01 = ds01[I_validate:]
    test_ds10 = ds10[I_validate:]
    test_ds02 = ds02[I_validate:]
    test_ds20 = ds20[I_validate:]
    
    print('train {} | validate {} | test {}'.format(len(train_ins1), len(validate_ins1), len(test_ins1))) 

    dataset = {
        'train': (train_ins1, train_ins2, train_ds12, train_ds21, train_ds01, train_ds10, train_ds02, train_ds20),
        'validate': (validate_ins1, validate_ins2, validate_ds12, validate_ds21, validate_ds01, validate_ds10, validate_ds02, validate_ds20),
        'test': (test_ins1, test_ins2, test_ds12, test_ds21, test_ds01, test_ds10, test_ds02, test_ds20),
    }
    result = (goal, dataset)
    
    return result


