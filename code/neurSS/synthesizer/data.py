import torch 
from torch.utils.data import Dataset
import os 
import json 
from code.utils.utils import *
from code.utils.parser.utils import *
import pickle 
import logging 

PAD_IDX = 0
START_IDX = 1
END_IDX = 2

class ProgramData(Dataset):

    def __init__(self, path_to_dataset, data_info, num_io = None, rotate = None, override = False,  num_load = None):


        logging.info(f'Loading data from :{path_to_dataset}')

        self.num_io = num_io 

        self.rotate = rotate 
            
        self.pad_idx = data_info["pad_idx"] 
        self.start_idx = data_info["start_idx"]
        self.end_idx =  data_info["end_idx"]

        self.IMG_SIZE = data_info["IMG_SIZE"]
        self.IMG_FEAT = data_info["IMG_FEAT"]
        self.grid_size = data_info["grid_size"]

        self.vocab = data_info["vocab"]
        self.vocab_size = data_info["vocab_size"]

        tkn2idx = data_info["vocab"]["tkn2idx"]
        path_to_ds_cache = path_to_dataset.replace('.data', '.thdump')

        if not override and os.path.exists(path_to_ds_cache):
            logging.info('Found cached file, loading pre-processed data')
            with open(path_to_ds_cache,'rb') as handle : 
                dataset = pickle.load(handle)

        else:    
            logging.info('Start data processing')
            with open(path_to_dataset, 'r') as f_dataset:
                dataset = f_dataset.readlines()

            input_grids = []
            output_grids = []
            target_seqs = []

            for program_idx, program_data in enumerate(dataset):
                # Get the target program
                program_data = json.loads(program_data)
                program_tokens = program_data['program_tokens']
                program_idxs = list(map(tkn2idx.get, program_tokens))
                
                input_grids_for_program = []
                output_grids_for_program = []
                # for each example 
                for grid_io in program_data['examples']:

                    inp_grid = grid_io['inpgrid_tensor']
                    out_grid = grid_io['outgrid_tensor']
                    inp_grid = torch.ShortTensor(inp_grid)
                    out_grid = torch.ShortTensor(out_grid)
                    inp_grid = grid_desc_to_tensor(inp_grid, self.IMG_FEAT, self.IMG_SIZE)
                    out_grid = grid_desc_to_tensor(out_grid, self.IMG_FEAT, self.IMG_SIZE)
                    input_grids_for_program.append(inp_grid)
                    output_grids_for_program.append(out_grid)

                input_grids_for_program = torch.stack(input_grids_for_program)
                output_grids_for_program = torch.stack(output_grids_for_program)

                input_grids.append(input_grids_for_program)
                output_grids.append(output_grids_for_program)
                target_seqs.append(program_idxs)

                if (num_load is not None) :
                    if (program_idx == num_load - 1) : break 

            input_grids = torch.stack(input_grids)
            output_grids = torch.stack(output_grids)

            input_lines = [torch.tensor([self.start_idx] + line) for line in target_seqs]
            input_lines_padded = torch.nn.utils.rnn.pad_sequence(input_lines, batch_first = True)
            output_lines_padded =  torch.nn.ConstantPad1d((0,1),0)(input_lines_padded[:,1:])

            logging.info(f'Loaded {len(input_lines)} data samples')

            # save max sequence length 
            max_len = input_lines_padded.shape[1]
            self.max_len = max_len 

            dataset = {
                "input_lines": input_lines_padded,
                "output_lines": output_lines_padded,
                "input_grids" : input_grids,
                "output_grids" : output_grids,
                "max_len"   : max_len
            }
            
            with open(path_to_ds_cache,'wb') as handle : 
                pickle.dump(dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)

        if num_load is not None : 
            self.input_lines = dataset["input_lines"][:num_load]
            self.output_lines = dataset["output_lines"][:num_load]
            self.input_grids = dataset["input_grids"][:num_load]
            self.output_grids = dataset["output_grids"][:num_load]
        else : 
            self.input_lines = dataset["input_lines"]
            self.output_lines = dataset["output_lines"]
            self.input_grids = dataset["input_grids"]
            self.output_grids = dataset["output_grids"]
        
        if num_io is not None : 
            self.input_grids =  self.input_grids[:,:self.num_io]
            self.output_grids =  self.output_grids[:,:self.num_io]

        if self.rotate : 
            assert self.input_grids.shape[1] == 1
            self.input_grids = get_all_grid_rotations(self.input_grids)
            self.output_grids = get_all_grid_rotations(self.output_grids)
    
        #test(self.input_grids,  self.output_grids, self.input_lines, self.vocab, idx  = 1)




    def __len__(self):
        return len(self.input_lines)
    

    def __getitem__(self, idx):
        return self.input_grids[idx], self.output_grids[idx],\
            self.input_lines[idx],self.output_lines[idx] 



def grid_desc_to_tensor(grid_desc, IMG_FEAT, IMG_SIZE):

    grid = torch.Tensor(IMG_FEAT).fill_(0).cpu()
    grid.index_fill_(0, grid_desc.long(), 1)
    grid = grid.view(IMG_SIZE)

    return grid

def load_generated_data_info(path_to_info):
    ''' 
        load dataset info into namespace object  
    '''
    data_info = {}
    data_info["pad_idx"] = PAD_IDX
    data_info["start_idx"] = START_IDX
    data_info["end_idx"] = END_IDX

    tkn2idx = {
        '<pad>': data_info["pad_idx"],
        '<s>'  : data_info["start_idx"],
        'm)'   : data_info['end_idx']
    }

    with open(path_to_info, 'r') as info_file:
        info = json.load(info_file)
    
    NEXT_ID = 3

    data_info["IMG_FEAT"] = info['img_feat']
    data_info["IMG_SIZE"] = info['img_size']
    data_info["grid_size"] = info['grid_size']

    tokens = info['tokens']
    token_idx = 0
    for token in tokens:
        if token not in tkn2idx :   
             tkn2idx[token] = token_idx + NEXT_ID
             token_idx +=1
    idx2tkn = { v : k for k,v in tkn2idx.items()}

    data_info["vocab"] = {
        "idx2tkn" : idx2tkn, 
        "tkn2idx" : tkn2idx
    }    
    data_info["vocab_size"] = len(idx2tkn) 
    return data_info




def get_all_grid_rotations(grids):
    
    device = grids.device
    
    rotated = []
    marked_cell_idxs = torch.where(grids)[2] 
    agent_idxs = marked_cell_idxs[marked_cell_idxs <= 3]
    agent_idxs_rotated = agent_idxs.clone()

    for rotation in range(4) : 

        rotated_grids = torch.rot90(grids, k=rotation + 1,  dims = (-1,-2) ).long()
        rotated_idxs = torch.where(rotated_grids)
        rotated_marked_cell_idxs = rotated_idxs[2]
        rotated_agent_y_pos = rotated_idxs[-1][rotated_marked_cell_idxs <= 3]
        rotated_agent_x_pos = rotated_idxs[-2][rotated_marked_cell_idxs <= 3]
        rotated_agent_idxs = rotated_marked_cell_idxs[rotated_marked_cell_idxs <= 3]

        grid_index = range(rotated_grids.shape[0])
        rotated_grids[grid_index, : , rotated_agent_idxs] = 0 
        
        agent_old_idxs = agent_idxs_rotated.clone()

        agent_idxs_rotated[agent_old_idxs == AGENT_WEST] = AGENT_NORTH
        agent_idxs_rotated[agent_old_idxs == AGENT_NORTH] = AGENT_EAST
        agent_idxs_rotated[agent_old_idxs == AGENT_EAST] = AGENT_SOUTH
        agent_idxs_rotated[agent_old_idxs == AGENT_SOUTH] = AGENT_WEST

       
        rotated_grids[grid_index, : , agent_idxs_rotated,  rotated_agent_x_pos,   rotated_agent_y_pos] = 1        
        rotated_grids = rotated_grids.float()
       
        rotated.append(rotated_grids.squeeze(1)) 

    rotated = torch.stack(rotated).transpose(0, 1)

    rotated = rotated.to(device)

    return rotated
