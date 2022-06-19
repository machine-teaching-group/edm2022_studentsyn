import argparse
import json
import time
from pathlib import Path

from ..utilities.karel.dsl_karel import DslKarel
from ..utilities.karel.program_karel import ProgramKarel
from ..utilities.ast_to_code_converter import convert_ast_to_code
from ..utilities.utils import write_input_output_pair
from ..utilities.karel.environment_karel import EnvironmentKarel
from ..utilities.utils import WallCrashError, \
        AgentBacktrackingError, AgentOutOfBoundsError, ExecutionTimeoutError, \
        PickEmptyMarkerError, PutMaxMarkerError

# true probability of hitting front_is_clear
PROB_FRONT_IS_CLEAR = 0.5
# true probability of hitting left_is_clear
PROB_LEFT_IS_CLEAR = 0.5
# true probability of hitting right_is_clear
PROB_RIGHT_IS_CLEAR = 0.5
# true probability of hitting markers_present
PROB_MARKERS_PRESENT = 0.5


def symbolic_execution(input_code, env_type, env_type_execution, program, prob_front_is_clear, 
                        prob_left_is_clear, prob_right_is_clear, prob_markers_present):
    if( env_type == "karel" ):
        allow_backtracking = True
    else:
        allow_backtracking = False
    env = EnvironmentKarel(mode="inverse", coin_flips=None, world_size=(12, 12),
                prob_front_is_clear=prob_front_is_clear, prob_left_is_clear=prob_left_is_clear, 
                prob_right_is_clear=prob_right_is_clear, prob_markers_present=prob_markers_present, 
                prob_no_markers_present=1-prob_markers_present, allow_backtracking=allow_backtracking)

    # add a reference to the env object in the parser object
    program.dsl.parser.env = env
    # flush parser info before run
    program.dsl.parser.flush_hit_info()
    task_input_str = env.draw_array(no_print=True)
    agent_input_dir = env.agent_input_dir
    agent_input_loc = env.agent_input_loc
    # execute program on task
    try:
        env.inverse_run(program)
    except WallCrashError:
        return [{"crashed" : "WallCrashError"}]
    except AgentBacktrackingError:
        return [{"crashed" : "AgentBacktrackingError"}]
    except ExecutionTimeoutError:
        return [{"crashed" : "ExecutionTimeoutError"}]
    except PutMaxMarkerError:
        return [{"crashed" : "PutMaxMarkerError"}]
    except PickEmptyMarkerError:
        return [{"crashed" : "PickEmptyMarkerError"}]
    except AgentOutOfBoundsError:
        return [{"crashed" : "AgentOutOfBoundsError"}]

    task_output_str = env.draw_array(no_print=True)
    agent_output_dir = env.task.agent.facing
    agent_output_loc = env.task.agent.position
    state_sequence = env.state_sequence
    location_trace = env.location_trace
    locked_cell = env.locked_empty_cells.union( env.locked_wall_cells.union( env.locked_marker_cells ) ) 
    input_marker = env.input_marker_cells
    locked_wall_cell = env.locked_wall_cells
    hit_info = program.dsl.parser.hit_info
    symbolic_decision = program.dsl.parser.symbolic_decisions
    # remove reference to env object in parser
    program.dsl.parser.env = None
    return [[task_input_str], [task_output_str], [state_sequence], [location_trace], 
                    [hit_info], [locked_cell], [agent_input_dir], [agent_input_loc], 
                    [agent_output_dir], [agent_output_loc], [input_marker], [locked_wall_cell], [symbolic_decision]]


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_out_path', type=str, default="./code/coderandom/output/", help='output folder path for new tasks')
    arg_parser.add_argument('--task_type', type=str, required=True, help='hoc/karel')
    arg_parser.add_argument('--input_program_path', type=str, required=True, help='filename of program as AST in json format')
    args = arg_parser.parse_args()

    # convert AST json file to code sequence
    with open(args.input_program_path) as f:
        input_ast = json.load(f)
    input_code = convert_ast_to_code(input_ast)

    # get environment type HOC / Karel
    env_type = args.task_type
    # HOC / Karel programs are executed as Karel programs
    env_type_execution = "karel"

    # execute single symbolic execution
    start_time_random = time.time()
    dsl = DslKarel()    
    program = ProgramKarel(input_code, dsl)
    out_list = symbolic_execution(input_code, env_type, env_type_execution, program,
                                    PROB_FRONT_IS_CLEAR, PROB_LEFT_IS_CLEAR, 
                                    PROB_RIGHT_IS_CLEAR, PROB_MARKERS_PRESENT)

    end_time_random = time.time()
    random_execution_time = round(end_time_random - start_time_random, 2)
    print("time taken to execute one symbolic execution: {}".format(random_execution_time))

    # write trace info as new task file
    if( len(out_list) == 1 ):
        print("failure: crashed due to {}".format(out_list[0]["crashed"]))
    else:
        # write trace info as new task file
        (task_inputs_str, task_outputs_str, state_sequences, location_traces, hit_infos, 
                locked_cells, agent_input_dirs, agent_input_locs, agent_output_dirs, agent_output_locs, 
                input_markers, locked_wall_cells, symbolic_decisions) = out_list   
        input_filename = Path(args.input_program_path).stem
        fname_prefix = input_filename + "_single-symbolic-exec"
        write_input_output_pair(task_inputs_str[0], task_outputs_str[0], 
                    state_sequences[0], location_traces[0], locked_cells[0],
                    agent_input_dirs[0], agent_input_locs[0], agent_output_dirs[0],
                    agent_output_locs[0], input_markers[0], locked_wall_cells[0],
                    fname_prefix, args.data_out_path, env_type, 0, args.input_program_path)
        print("success: new task saved")


if __name__ == '__main__':
    main()