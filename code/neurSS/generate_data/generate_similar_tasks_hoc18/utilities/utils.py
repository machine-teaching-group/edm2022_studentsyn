'''
code based on https://github.com/carpedm20/karel, 
and https://github.com/alts/karel
'''

import numpy as np
import random
import os
import errno
import signal
from functools import wraps
from pyparsing import nestedExpr


dir_name_to_tuple = {
    'east': (1, 0),
    'north': (0, -1),
    'west': (-1, 0),
    'south': (0, 1),
}

class CoinFlipMismatchError(Exception):
    pass


class ExceededPreFlippedCoinsError(Exception):
    pass

    
class UnknownMctsTreeTypeError(Exception):
    pass


class PutMaxMarkerError(Exception):
    pass


class PickEmptyMarkerError(Exception):
    pass


class WallCrashError(Exception):
    pass


class AgentBacktrackingError(Exception):
    pass


class AgentOutOfBoundsError(Exception):
    pass


class ExecutionTimeoutError(Exception):
    pass


def get_rng(rng, seed=None):
    if rng is None:
        rng = np.random.RandomState(seed)
    return rng


def get_hash():
    return random.getrandbits(128)


def dummy():
    pass


def str2bool(v):
    return v.lower() in ('true', '1')

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL,seconds) #used timer instead of alarm
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    return decorator


def beautify_fn(inputs, indent=1, tabspace=2):
    lines, queue = [], []
    space = tabspace * " "

    for item in inputs:
        if item == ";":
            lines.append(" ".join(queue))
            queue = []
        elif type(item) == str:
            queue.append(item)
        else:
            lines.append(" ".join(queue + ["{"]))
            queue = []

            inner_lines = beautify_fn(item, indent=indent+1, tabspace=tabspace)
            lines.extend([space + line for line in inner_lines[:-1]])
            lines.append(inner_lines[-1])

    if len(queue) > 0:
        lines.append(" ".join(queue))

    return lines + ["}"]


def pprint(code, *args, **kwargs):
    print(beautify(code, *args, **kwargs))


def beautify(code, tabspace=2):
    code = " ".join(replace_dict.get(token, token) for token in code.split())
    array = nestedExpr('{','}').parseString("{"+code+"}").asList()
    lines = beautify_fn(array[0])
    return "\n".join(lines[:-1]).replace(' ( ', '(').replace(' )', ')')


def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)


def get_direction_string(direction):
    if( direction == (-1, 0) ):
        return "west"
    elif( direction == (1, 0) ):
        return "east"
    elif( direction == (0, -1) ):
        return "north"
    else:
        return "south"


def write_input_output_pair(task_input_str, task_output_str, 
                        state_sequence, location_trace, locked_cell,
                        agent_input_dir, agent_input_loc, agent_output_dir,
                        agent_output_loc, input_marker, locked_wall_cell,
                        filename_prefix, directory, env_type, temp, 
                        program_path):
    gridsz = 12
    if( temp == 1 ):
        out_filename = filename_prefix + "_task.txt"
        task_path = os.path.join(directory, "temp")
        filename = os.path.join(task_path, out_filename)
    elif( temp == 3 ):
        out_filename = filename_prefix + "_task.txt"
        filename = os.path.join(directory, out_filename)
    else:
        # copy code file - using cp from linux is fastest (assumption) over python copying
        #directory = os.path.join(directory, "tasks")
        #program_out_path = os.path.join(directory, filename_prefix + "_code.json")
        #subprocess.call("cp {} {}".format(program_path, program_out_path), shell=True)

        out_filename = filename_prefix + "_task-out.txt"
        filename = os.path.join(directory, out_filename)

    if( env_type == "hoc" ):
        # fill surrounding of output grid with wall cells
        all_cells = set()
        for i in range(gridsz):
            for j in range(gridsz):
                all_cells.add(str(i) + '#' + str(j))
        surrounding_wall_cells = all_cells.difference(locked_cell)
        for pos in surrounding_wall_cells:
            i = int(pos.split("#")[0])
            j = int(pos.split("#")[1])
            task_output_str[i][j] = "#"     

        # add goal to output grid (since some tasks don't have a while(no goal))
        x = agent_output_loc[0] 
        y = agent_output_loc[1]
        task_output_str[y][x] = "+"

        # start writing to file
        write_string = ""
        with open(filename, "w") as f:      
            write_string += "type\thoc\n"
            write_string += "gridsz\t" + "(" + str(gridsz) + "," + str(gridsz) + ")" + "\n"
            write_string += "\n"
            write_string += "pregrid"
            for i in range(1, gridsz+1):
                write_string += "\t" + str(i)
            write_string += "\n"

            task_output_str = ["\t".join(row.tolist()) for row in task_output_str]
            for i in range(1, len(task_output_str)+1):
                write_string += str(i) + "\t"
                write_string += task_output_str[i-1]
                write_string += "\n"

            x = agent_input_loc[0] + 1
            y = agent_input_loc[1] + 1
            write_string += "agentloc\t" + "(" + str(x) + "," + str(y) + ")" + "\n"

            agent_input_dir = get_direction_string(agent_input_dir)
            write_string += "agentdir\t" + agent_input_dir + "\n"

            f.write(write_string)

    else:
        # place input markers in input grid
        for pos in input_marker:
            i = int(pos.split("#")[0])
            j = int(pos.split("#")[1])
            task_input_str[i][j] = "x"
        # place walls in input grid
        for pos in locked_wall_cell:
            i = int(pos.split("#")[0])
            j = int(pos.split("#")[1])
            task_input_str[i][j] = "#"
        
        # start writing to file
        write_string = ""
        with open(filename, "w") as f:      
            write_string += "type\tkarel\n"
            write_string += "gridsz\t" + "(" + str(gridsz) + "," + str(gridsz) + ")" + "\n"
            write_string += "\n"

            # write pregrid
            write_string += "pregrid\t"
            for i in range(1, gridsz+1):
                write_string += str(i) + "\t"
            write_string += "\n"

            task_input_str = ["\t".join(row.tolist()) for row in task_input_str]
            for i in range(1, len(task_input_str)+1):
                write_string += str(i) + "\t"
                write_string += task_input_str[i-1]
                write_string += "\n"

            x = agent_input_loc[0] + 1
            y = agent_input_loc[1] + 1
            write_string += "agentloc\t" + "(" + str(x) + "," + str(y) + ")" + "\n"

            agent_input_dir = get_direction_string(agent_input_dir)
            write_string += "agentdir\t" + agent_input_dir + "\n"   
            write_string += "\n"

            # write postgrid    
            write_string += "postgrid\t"
            for i in range(1, gridsz+1):
                write_string += str(i) + "\t"
            write_string += "\n"

            task_output_str = ["\t".join(row.tolist()) for row in task_output_str]
            for i in range(1, len(task_output_str)+1):
                write_string += str(i) + "\t"
                write_string += task_output_str[i-1]
                write_string += "\n"

            x = agent_output_loc[0] + 1
            y = agent_output_loc[1] + 1
            write_string += "agentloc\t" + "(" + str(x) + "," + str(y) + ")" + "\n"

            agent_output_dir = get_direction_string(agent_output_dir)
            write_string += "agentdir\t" + agent_output_dir + "\n"

            f.write(write_string)


def get_input_task_data(env_type, task_path):
    if( env_type == "karel" ):
        with open(task_path) as f:
            lines = f.readlines()
        # pregrid
        pregrid = lines[4:16]
        pregrid = ["".join(line.strip().split("\t")[1:]) for line in pregrid]
        # postgrid
        postgrid = lines[20:32]
        postgrid = ["".join(line.strip().split("\t")[1:]) for line in postgrid]
        # agentloc 
        pos = lines[16].strip().split("\t")[1]
        x = int(pos.split(",")[0][1:]) - 1
        y = int(pos.split(",")[1][:-1]) - 1
        agent_input_loc = (x+1, y+1)
        # agentdir
        dir_name = lines[17].strip().split("\t")[1]
        agent_input_dir = dir_name_to_tuple[dir_name]
        
        input_task_data = {"agent_input_loc" : agent_input_loc,
                            "agent_input_dir" : agent_input_dir,
                            "pregrid" : pregrid,
                            "postgrid" : postgrid,
                            "input_gridsz" : 12
                            }
    else:
        with open(task_path) as f:
            read_flag = False
            lines = []
            for line in f.readlines():
                if("pregrid" in line):
                    read_flag = True
                    continue
                if( read_flag == True ):
                    if( "agentloc" in line ):
                        pos = line.strip().split("\t")[1]
                        x = int(pos.split(",")[0][1:]) - 1
                        y = int(pos.split(",")[1][:-1]) - 1
                        agent_input_loc = (x+1, y+1)
                    elif( "agentdir" in line ):
                        dir_name = line.strip().split("\t")[1]
                        agent_input_dir = dir_name_to_tuple[dir_name]
                    else:   
                        line = "".join(line.strip().split("\t")[1:])
                        lines.append(line)
        
        input_task_data = {"agent_input_loc" : agent_input_loc,
                            "agent_input_dir" : agent_input_dir,
                            "pregrid" : lines,
                            # input task gridsz is always 12 for now
                            "input_gridsz" : 12
                            }

    return input_task_data

