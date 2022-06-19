import time
import numpy as np 
from copy import deepcopy 
from code.utils.utils import * 
from code.utils.parser.world import World 
import logging 

from .utils_gen_data import env_type_execution, env_type, facing2dir, facing2idx, mutate_grid
from .utils_gen_data import EPSILON, PROB_FRONT_IS_CLEAR, PROB_LEFT_IS_CLEAR, PROB_RIGHT_IS_CLEAR, PROB_MARKERS_PRESENT

from .generate_similar_tasks_hoc18.utilities.karel.dsl_karel import DslKarel
from .generate_similar_tasks_hoc18.utilities.karel.program_karel import ProgramKarel
from .generate_similar_tasks_hoc18.utilities.get_num_blocks_code import get_num_blocks
from .generate_similar_tasks_hoc18.utilities.utils_shortest_path_on_grid import prune_using_shortest_path_hoc
from .generate_similar_tasks_hoc18.coderandom.run_one_symbolic_execution import symbolic_execution
from .generate_similar_tasks_hoc18.utilities.scoring import compute_score




def generate_grids_hoc18(candidate_code, grid_size, random_generator, 
    number_IO, patience, num_iterations, debug = False, initialize_grid = 'blocks', 
    WALL_CHAR = '#', EMPTY_CHAR = '.'):

    world_obj = World()
    world_obj.create_random_empty(grid_size, random_generator)

    if initialize_grid == 'blocks' : 
        world_obj.world = np.zeros_like(world_obj.world)
        world_obj.world.fill(World.WALL_CHAR)
 
    elif initialize_grid == 'random':
        world_obj.world = random_generator.choice([WALL_CHAR,EMPTY_CHAR],world_obj.world.shape)

    pregrid = []
    for data_row in world_obj.world.tolist():
        pregrid.append(''.join(data_row))

    task_data = {
        "agent_input_loc" : world_obj.agent.position,
        "agent_input_dir" : world_obj.agent.facing,
        "pregrid" : pregrid,
        "input_gridsz" : world_obj.grid_size
        }
 
    task_traces = generate_grids_for_code(candidate_code, task_data, number_IO, 
        patience, num_iterations, debug = debug)
    
    examples = []
    if task_traces != None : 
        for mcmc_trace in task_traces : 
            input_tensor, output_tensor, exec_trace = run_generated_mcts(mcmc_trace, candidate_code, 
                random_generator, grid_size, debug = debug)

            examples.append({        
                "example_index" : len(examples)+1,
                "inpgrid_tensor" : input_tensor,
                "outgrid_tensor" : output_tensor
            })

    return examples 



def generate_grids_for_code(input_code, input_task_data,   number_IO, patience, num_iterations,
    use_single_rollout = False, debug = False):    

    num_wrong_exec = 0 
    generated_grids = []
    if use_single_rollout : 
        generated_grids = get_traces(input_code, input_task_data, generated_grids, number_IO, num_iterations)
        return generated_grids

    num_io_generated = 0 

    while num_io_generated < number_IO:

        max_mcts_trace = get_traces(input_code, input_task_data, generated_grids, 1, num_iterations, debug = debug)

        if max_mcts_trace == None : 

            num_wrong_exec += 1            
            if num_wrong_exec > patience : 
                return []

        else : 

            num_wrong_exec = 0
            generated_grids.append(max_mcts_trace[0])
            num_io_generated += 1

    return generated_grids





def get_traces(input_code, input_task_data, Z, top_k, num_iterations, debug = False):

    start_time_random = time.time()
    dsl = DslKarel()        
    program = ProgramKarel(input_code, dsl)

    all_traces = execute_random_rollouts( 
                        num_iterations, input_code, input_task_data, 
                        env_type_execution, env_type, Z, program,
                        PROB_FRONT_IS_CLEAR, PROB_LEFT_IS_CLEAR, 
                        PROB_RIGHT_IS_CLEAR, PROB_MARKERS_PRESENT)
    end_time_random = time.time()
    random_execution_time = round(end_time_random - start_time_random, 2)
    if debug : logging.info("time taken to execute {} random rollouts: {}".format(num_iterations, random_execution_time))

    # pick top trace from pool of mctget_tracess traces seen during training after pruning
    start_time_pick_trace = time.time()
    max_mcts_trace = pick_top_trace_with_pruning(all_traces, input_code, top_k, debug = debug)    
    end_time_pick_trace = time.time()
    mcts_tree_pick_trace_time = round(end_time_pick_trace - start_time_pick_trace, 2)

    if debug : logging.info(f'time taken to pick trace: {mcts_tree_pick_trace_time}')

    return max_mcts_trace

def pick_top_trace_with_pruning(all_traces,input_code, top_k, debug = False):

    all_traces.sort(key=lambda trace: trace[-1]["score_total"], reverse=True)
    return_traces = []
    past_tasks = set()
    for trace in all_traces:
        # if crashed
        if( "crashed" in trace[-1] ):
            continue
        # if coverage != 1
        elif( trace[-1]["score_coverage"] != 1 ):
            continue
        # filter invalid traces marked with 0.0 total_score (repeat traces in diversity and repeating input trace)
        elif( trace[-1]["score_total"] < EPSILON ):
            continue
        task_str = trace[0][1][0].tostring()
        if task_str in past_tasks : 
            continue 
        
        # TODO compute maxnumblocks and perform minimality check using graph shortest path
        # else prune using graph shortest path and check minimal
        else:
            # compare number of core blocks with basic actions minimal solution
            maxnumblocks = get_num_blocks(input_code)

            # prune_using_shortest_path_hoc - minimality is true (and no pruning) if p_in code with maxnumblocks is minimal code, 
            # false (and pruning) if shorter length code is found          
            task_data = prepare_task_data(trace)
            result = prune_using_shortest_path_hoc(task_data, maxnumblocks)
            if( result == "lesser" ):  
                continue
            else :
                return_traces.append(trace[0])
                past_tasks.add(task_str)

            if len(return_traces) == top_k : 
                if debug : 
                    logging.info(f'Found {len(return_traces)}/{top_k} for code :\n\t{input_code}')
                return return_traces

    if debug :
        logging.info(f'Found {len(return_traces)}/{top_k} tasks')
    
    return None 


def prepare_task_data(trace):
    '''
        format test data for `prune_using_shortest_path_hoc()'
    '''    

    trace_copy = deepcopy(trace[0])
    (task_inputs_str, task_outputs_str, state_sequences, location_traces, 
                hit_infos, locked_cells, agent_input_dirs, agent_input_locs, agent_output_dirs, 
                agent_output_locs, input_markers, locked_wall_cells, symbolic_decisions) = trace_copy

    grid_to_int_dict = {"#" : 1, "." : 0, "x" : 2}
    grid =  np.vectorize(grid_to_int_dict.get)(task_outputs_str[0])
    grid = grid.squeeze()  
    
    sz = grid.shape
    start = agent_input_locs[0]
    start = (start[1],start[0])
    end = agent_output_locs
    end[0] = (end[0][1],end[0][0])
    direction = facing2dir[agent_input_dirs[0]]
    
    task_data = (env_type, sz, start, direction, end, grid)
    return task_data 


def get_world_obj_from_mcmc_trace(trace, grid_size):
    
    
    (task_inputs_str, task_outputs_str, state_sequences, location_traces, 
                hit_infos, locked_cells, agent_input_dirs, agent_input_locs, agent_output_dirs, 
                agent_output_locs, input_markers, locked_wall_cells, symbolic_decisions) = trace
    
    
    world_obj = World()
    world_obj.create_random_empty(grid_size)
    world_obj.world = deepcopy(task_outputs_str[0])

    markers = np.where(world_obj.world == 'x')
    world_obj.world[markers] = '.'
 
    agent_pos_x,agent_pos_y = agent_input_locs[0]
    world_obj.agent.position =  (agent_pos_x, agent_pos_y)
    world_obj.agent.facing = agent_input_dirs[0]

    world_obj.goal = agent_output_locs[0][1], agent_output_locs[0][0]
    world_obj.world[world_obj.goal] = '.'
    world_obj.goal_set = True 
    world_obj.init_pos = (agent_pos_x, agent_pos_y, facing2idx[world_obj.agent.facing])  
        
    
    return world_obj

def run_generated_mcts(mcmc_trace, candidate_code, random_generator, grid_size, debug = False):


    world_obj = get_world_obj_from_mcmc_trace(mcmc_trace, grid_size)
    world_obj.run(candidate_code)
    if debug : 
        world_obj.print(print_fn = logging.info)
    exec_trace = world_obj.trace
    grid = mutate_grid_hoc18(exec_trace, mcmc_trace, world_obj, candidate_code, random_generator)

    world_obj.world = grid

    if debug : 
        logging.info('world after mutation')
        world_obj.print(print_fn  = logging.info)

    w_init = deepcopy(world_obj)    
    w_init.agent.set_position(*world_obj.init_pos)    
    input_grid = w_init.get_idxs()
    output_grid = world_obj.get_idxs()

    return input_grid, output_grid, exec_trace



def mutate_grid_hoc18(exec_trace, mcmc_trace, world_obj_orig, input_code, random_generator, prob_rand_line = 0.1, prob_scatter = 0.1, debug = False):
    
    grid = world_obj_orig.world
    grid_size = grid.shape[-1]
    generated_grids = mutate_grid(grid, random_generator, grid_size, prob_rand_line = prob_rand_line, 
        prob_scatter = prob_scatter, scatter_type = 'variant_2')
    
    for gen_grid in generated_grids[::-1]:
        task_data = prepare_task_data([mcmc_trace])
        task_data = list(task_data)
        task_data[-1] = gen_grid
        task_data = tuple(task_data)

        maxnumblocks = get_num_blocks(input_code)            
        result = prune_using_shortest_path_hoc(task_data, maxnumblocks)
        if result == 'lesser' : continue 

        world_obj_mutated = get_world_obj_from_mcmc_trace(mcmc_trace, grid_size)
        world_obj_mutated.world = gen_grid
        world_obj_mutated.run(input_code,debug = False)
        exec_trace_mutated = world_obj_mutated.trace 

        if exec_trace_mutated == exec_trace : 
            return gen_grid 

    return grid


def score_rollout(out_list, input_task_data, env_type, Z):
    
    if( len(out_list) == 1 ):
        score_dict = {
            "score_total" : 0.0,
            "crashed": True,
            "crash_type" : out_list[0]["crashed"]
        }
        return score_dict
    else:
        score_dict = compute_score(out_list, input_task_data, env_type, Z, tree_type="NA", gridsz=12)    
        return score_dict


def execute_random_rollouts(num_iterations, input_code, input_task_data, 
                                env_type_execution, env_type, Z, program,
                                prob_front_is_clear, prob_left_is_clear, 
                                prob_right_is_clear, prob_markers_present):
    all_traces = []
    # execute num_iterations random rollouts
    start_time = time.time()
    for _ in range(num_iterations):
        # execute single symbolic execution
        
        out_list = symbolic_execution(input_code, env_type, env_type_execution, program,
                                        prob_front_is_clear, prob_left_is_clear, 
                                        prob_right_is_clear, prob_markers_present)
        # compute score of rollout
        score_dict = score_rollout(out_list, input_task_data, env_type, Z)
        # store rollout with score
        all_traces.append((out_list, score_dict))
        elapsed_time = time.time() - start_time
        if elapsed_time > 1 : 
            break 

    return all_traces