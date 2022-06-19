from .utils import parse_asts
from code.utils.utils import EMB_DIR
from code.embeddings.embedding_util import EmbUtil
from code.neurSS.synthesizer.data import get_all_grid_rotations
from code.utils.convert import benchmark_code_to_student_code

import pickle
import os 
import logging 

import numpy as np 
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

class TrainingDataUtil(Dataset):
    def __init__(self, source_task, data_info, input_grid, output_grid, embedding_models_dir):
        self.source_task = source_task
        self.data_info = data_info
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.embedding_models_dir = embedding_models_dir

        assert self.input_grid.shape[1] == 1
        self.input_grid = get_all_grid_rotations(self.input_grid)
        self.output_grid = get_all_grid_rotations(self.output_grid)
        self.load_neighbor_data()

    def load_neighbor_data(self):

        student_embeddings_path = EMB_DIR + f'/{self.source_task}/{self.embedding_models_dir}/{self.source_task}_data_embedded.p'

        logging.info(f'loading embedded student data  from {student_embeddings_path}')

        if not os.path.exists(student_embeddings_path):
            raise  Exception('could not find embedded student data')

        with open(student_embeddings_path, 'rb') as handle : 
            student_data = pickle.load(handle)
        
        embeddings = student_data["embeddings"]
        asts = student_data["asts"]

        emb_util = EmbUtil(self.source_task, self.embedding_models_dir)    
        neighbors_obj = NearestNeighbors(p = 2)
        neighbors_obj.fit(embeddings)

        self.embeddings = embeddings

        self.asts_array = np.array(asts, dtype = object) 
        self.neighbors_obj = neighbors_obj
        self.emb_util = emb_util

    def select_neighbors(self, code_A, r):
        code_A_converted = benchmark_code_to_student_code(code_A)    
        embedding_A = self.emb_util.get_emb([code_A_converted])

        distances, student_idxs = self.neighbors_obj.radius_neighbors(
                embedding_A.reshape(1,-1),
                radius = r)

        student_idxs = student_idxs[0]
    
        neighbor_asts = self.asts_array[student_idxs].tolist()
        neighbor_asts += [code_A_converted]

        self.distances = distances        

        self.neighbor_in_tokens, self.neighbor_out_tokens = parse_asts(neighbor_asts, self.data_info)

        number_train_points = len(self.neighbor_in_tokens)
        self.neighbor_in_grids = self.input_grid.repeat(number_train_points, 1, 1, 1, 1)
        self.neighbor_out_grids = self.output_grid.repeat(number_train_points, 1, 1, 1, 1)

        print(f'found {number_train_points-1} neighbors')

        self.number_train_points = number_train_points


    def __len__(self):
        return self.number_train_points

    def __getitem__(self, idx):
        return self.neighbor_in_grids[idx], self.neighbor_out_grids[idx],\
            self.neighbor_in_tokens[idx],self.neighbor_out_tokens[idx] 
