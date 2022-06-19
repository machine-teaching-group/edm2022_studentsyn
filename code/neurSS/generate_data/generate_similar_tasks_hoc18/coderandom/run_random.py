import argparse
import os
import json
import time
import random
import string
from pathlib import Path

from ..codemcts.utilities.ast_to_code_converter import convert_ast_to_code
from ..codemcts.utilities._utils_shortest_path_on_grid import prune_using_shortest_path_hoc
from ..codemcts.utilities.karel.dsl_karel import DslKarel
from ..codemcts.utilities.karel.program_karel import ProgramKarel
from ..codemcts.utilities.utils import write_input_output_pair, get_input_task_data
from ..codemcts.utilities._get_num_blocks_code import get_num_blocks
from ...random_algorithm import execute_random_rollouts
from ..codemcts.utilities._utils_shortest_path_karel import prune_using_shortest_path_karel

# true probability of hitting front_is_clear
PROB_FRONT_IS_CLEAR = 0.5
# true probability of hitting left_is_clear
PROB_LEFT_IS_CLEAR = 0.5
# true probability of hitting right_is_clear
PROB_RIGHT_IS_CLEAR = 0.5
# true probability of hitting markers_present
PROB_MARKERS_PRESENT = 0.5
EPSILON = 1e-6


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_out_path', type=str, default="./code/coderandom/output/", help='output folder path for new tasks')
    arg_parser.add_argument('--input_task_filename', type=str, required=True, help='filename of input task')
    arg_parser.add_argument('--input_program_path', type=str, required=True, help='filename of program as AST in json format')
    arg_parser.add_argument('--num_diverse_tasks', type=int, default=10, help='number of diverse new tasks to generate for input task')
    arg_parser.add_argument('--num_iterations', type=int, default=10000, help='number of random rollouts to execute')
    args = arg_parser.parse_args()

    task_path = os.path.join("./code/coderandom/input/", "{}".format(args.input_task_filename))
    with open(task_path) as f:
        input_task = f.read()
    # convert AST json file to code sequence
    with open(args.input_program_path) as f:
        input_ast = json.load(f)
    input_code = convert_ast_to_code(input_ast)

    # get environment type HOC / Karel
    if( "hoc" in input_task ):
        env_type = "hoc"
    else:
        env_type = "karel" 
    # HOC / Karel programs are executed as Karel programs
    env_type_execution = "karel"

    # get input task data
    input_task_data = get_input_task_data(env_type, task_path)

    # set Z stores the set of new tasks picked
    Z = []
    for i in range(args.num_diverse_tasks):
        print("getting diverse task: ", i+1)

        # execute args.num_iterations random symbolic executions / rollouts
        start_time_random = time.time()
        # get dsl and program object at the beginning to share across random rollouts
        dsl = DslKarel()    
        program = ProgramKarel(input_code, dsl)
        all_traces = execute_random_rollouts( 
                                args.num_iterations, input_code, input_task_data, 
                                env_type_execution, env_type, Z, program,
                                PROB_FRONT_IS_CLEAR, PROB_LEFT_IS_CLEAR, 
                                PROB_RIGHT_IS_CLEAR, PROB_MARKERS_PRESENT)
        end_time_random = time.time()
        random_execution_time = round(end_time_random - start_time_random, 2)
        print("time taken to execute {} random rollouts: {}".format(args.num_iterations, random_execution_time))

        # pick top trace from pool of mcts traces seen during training after pruning
        start_time_pick_trace = time.time()
        max_mcts_trace = pick_top_trace_with_pruning(all_traces, args.input_program_path, args.data_out_path, 
                                    env_type, input_code, env_type_execution, program)
        end_time_pick_trace = time.time()
        mcts_tree_pick_trace_time = round(end_time_pick_trace - start_time_pick_trace, 2)
        print("time taken to pick trace: ", mcts_tree_pick_trace_time)

        # finish post processing of current run
        if( max_mcts_trace == None ):
            print("no new task found")
            break
        else:
            # write trace info as new task file
            (task_inputs_str, task_outputs_str, state_sequences, location_traces, hit_infos, 
                    locked_cells, agent_input_dirs, agent_input_locs, agent_output_dirs, agent_output_locs, 
                    input_markers, locked_wall_cells, symbolic_decisions) = max_mcts_trace[0]  
            input_task_filename = Path(args.input_task_filename).stem
            fname_prefix = input_task_filename + "_random-diverse-{}".format(i+1)   
            write_input_output_pair(task_inputs_str[0], task_outputs_str[0], 
                        state_sequences[0], location_traces[0], locked_cells[0],
                        agent_input_dirs[0], agent_input_locs[0], agent_output_dirs[0],
                        agent_output_locs[0], input_markers[0], locked_wall_cells[0],
                        fname_prefix, args.data_out_path, env_type, 0, args.input_program_path)

            # update set of picked traces
            Z.append(max_mcts_trace[0])


def pick_top_trace_with_pruning(all_traces, code_path, data_out_path, 
                                    env_type, input_code, env_type_execution, program):
    all_traces.sort(key=lambda trace: trace[-1]["score_total"], reverse=True)
    # since the temp folder is shared, to prevent parallel process to access the same file, adding a random string
    # useful when multiple run.py parallel invocations are performed
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=64))
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
        # else prune using graph shortest path and check minimal
        else:
            maxnumblocks = get_num_blocks(input_code)
            # write temp task file for trace
            if( env_type == "hoc" ):
                filename_prefix = "temp_hoc_" + random_str
                out_filename = "temp_hoc_{}_task.txt".format(random_str)
            else:
                filename_prefix = "temp_karel_" + random_str
                out_filename = "temp_karel_{}_task.txt".format(random_str) 
            (task_inputs_str, task_outputs_str, state_sequences, location_traces, 
                        hit_infos, locked_cells, agent_input_dirs, agent_input_locs, agent_output_dirs, 
                        agent_output_locs, input_markers, locked_wall_cells, symbolic_decisions) = trace[0]
            write_input_output_pair(task_inputs_str[0], task_outputs_str[0], 
                        state_sequences[0], location_traces[0], locked_cells[0],
                        agent_input_dirs[0], agent_input_locs[0], agent_output_dirs[0],
                        agent_output_locs[0], input_markers[0], locked_wall_cells[0],
                        filename_prefix, data_out_path, env_type, 1, code_path)
            task_path = os.path.join(data_out_path, "temp")
            task_path = os.path.join(task_path, out_filename)

            # prune_using_shortest_path_hoc - minimality is true (and no pruning) if p_in code with maxnumblocks is minimal code, 
            # false (and pruning) if shorter length code is found
            if( env_type == "karel" ):
                result = prune_using_shortest_path_karel(task_path, maxnumblocks)
            else:
                result = prune_using_shortest_path_hoc(task_path, maxnumblocks)
            
            if( result == "lesser" ):
                    continue
            else:
                return trace

    return None


if __name__ == '__main__':
    main()



