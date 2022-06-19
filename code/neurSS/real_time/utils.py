import torch 
from code.utils.parser.world import World
from code.utils.convert import ast_code_to_token_list

def parse_asts(asts, data_info):
    input_lines = []

    for ast in asts : 
        code_tokens = ast_code_to_token_list(ast)
        solution_idxs = list(map(data_info["vocab"]["tkn2idx"].get, code_tokens))
        program_tensor = torch.tensor([data_info["start_idx"]] + solution_idxs)
        input_lines.append(program_tensor)

    input_lines = torch.nn.utils.rnn.pad_sequence(input_lines, batch_first = True)
    output_lines = torch.nn.ConstantPad1d((0,1),0)(input_lines[:,1:]) 

    return input_lines, output_lines


def parse_task(task_info, data_info):

    world_obj = World()
    world_obj.read_from_file(lines = task_info["task"].split('\n'))    
    inp_grid = world_obj.get_state()
    inp_grid = torch.tensor(inp_grid).unsqueeze(0).unsqueeze(0).float()
    solution_tokens = ast_code_to_token_list(task_info["solution"])
    solution_str = ' '.join(solution_tokens)
    world_obj.run(solution_str)
    out_grid = world_obj.get_state()
    out_grid = torch.tensor(out_grid).unsqueeze(0).unsqueeze(0).float()
    
    solution_idxs = list(map(data_info["vocab"]["tkn2idx"].get, solution_tokens))
    input_lines = torch.tensor([data_info["start_idx"]] + solution_idxs).unsqueeze(0)
    output_lines =  torch.nn.ConstantPad1d((0,1),0)(input_lines[:,1:])

    return inp_grid, out_grid, input_lines, output_lines

def score_prob(model, inp_grids, out_grids, in_tgt_seq, device = 'cuda'):
   with torch.no_grad():
        model = model.to(device)

        inps = inp_grids.to(device)
        outs = out_grids.to(device)
        code = in_tgt_seq.to(device)
        # parse codes over model
        output, _, _, _ = model(inps, outs, code)
        # get log_prob of output
        logsoftmax = torch.nn.LogSoftmax(dim=-1)
        log_probs = logsoftmax(output)
        
        # ignore <s> and pad once at the end, 
        # create desired output sequence            
        c_out = torch.nn.ConstantPad1d((0,1),0)(code[:,1:])
        
        # ignore <pad> in probabilitiesut.
        weights = torch.ones_like(c_out)
        weights[torch.where(c_out == 0)] = 0 

        # index each N x max_len with correct index 
        p = log_probs.gather(-1,c_out.unsqueeze(-1))
        p = p.squeeze() * weights 
        lens = torch.sum(weights, dim = -1)
        # sum probabilities over each sequence 
        probs_raw = torch.sum(p, axis = - 1)
        sum_probs = probs_raw / lens 

        return  sum_probs, p