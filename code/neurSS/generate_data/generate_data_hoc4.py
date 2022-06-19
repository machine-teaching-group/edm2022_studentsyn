import numpy as np 
from copy import deepcopy
import networkx as nx 
import itertools 
import logging

from code.utils.parser.world import World 
from .utils_gen_data import mutate_grid

def generate_grids_hoc4(candidate_code, random_generator, number_IO, grid_size, patience,
    debug = False):

    node_to_idx, idx_to_node = get_nodes(grid_size)
    
    gen_examples = []
    num_wrong_exec = 0 
    if debug : 
        logging.info('Generating for code:')
        logging.info('\t' + candidate_code)
            
    while len(gen_examples) < number_IO :
        if num_wrong_exec > patience : 
            return []

        world_obj = World()
        world_obj.goal = (-1,-1)
        world_obj.create_random_empty(grid_size = grid_size, rng = random_generator)
        world_obj.run(candidate_code, track_visit = True)

        if world_obj.looped : 
            if debug : logging.info('circular code')
            return []

        if world_obj.crashed : 
            if debug : logging.info('crashed')
            num_wrong_exec += 1 
            continue 
        

        exec_trace = world_obj.trace
        world_obj.world = np.zeros_like(world_obj.world)
        world_obj.world.fill(World.WALL_CHAR)

        grid_cells_visited = np.array(exec_trace)[:,:2]
        x_pos_visit = grid_cells_visited[:,0]
        y_pos_visit = grid_cells_visited[:,1]
        world_obj.world[x_pos_visit, y_pos_visit] = World.EMPTY_CHAR

        if debug :  
            logging.info(f'generated grid #{len(gen_examples)+1}')                
            logging.info(exec_trace)
            world_obj.print(print_fn = logging.info)

        grid = world_obj.world
        generated_grids = mutate_grid(grid, random_generator, grid_size)

        mutated_grid = generated_grids[-1]
        mutated_grid_is_minimal = check_minimal(exec_trace, mutated_grid, node_to_idx, idx_to_node)    

        if mutated_grid_is_minimal : 
            final_grid = mutated_grid
            final_is_minimal = True  
        else : 
            final_grid = grid
            final_is_minimal = check_minimal(exec_trace, grid, node_to_idx, idx_to_node)
 
        world_obj.world = final_grid
        if debug : 
            logging.info('World after mutation')
            world_mutated = deepcopy(world_obj)
            world_mutated.world = mutated_grid
            world_mutated.print(print_fn = logging.info)

        if not final_is_minimal : 
            if debug : logging.info('not minimal')
            return []
                
        num_wrong_exec = 0 
        init_world = deepcopy(world_obj)
        init_world.agent.set_position(*world_obj.init_pos)
        input_grid = init_world.get_idxs()
        output_grid = world_obj.get_idxs()
        gen_examples.append({        
            "example_index" : len(gen_examples)+1,
            "inpgrid_tensor" : input_grid,
            "outgrid_tensor" : output_grid,
        })

    return gen_examples



def get_nodes(grid_size):
    nodes = itertools.product(range(grid_size),range(grid_size),range(4))
    node_to_idx = {n : i  for i,n in enumerate(nodes)}
    idx_to_node = {v : k for k,v in node_to_idx.items()}
    return node_to_idx, idx_to_node



def get_neighbors(node, grid, gridsize, WALL_CHAR = '#'):
    dirs = {
        1 : [3,2],
        3 : [0,1],
        0 : [3,2],
        2 : [0,1],
    }
    x_coord, y_coord, direction = node 
    neighbors = []
    if grid[x_coord, y_coord] == WALL_CHAR:
        return neighbors
    
    if direction == 1: # east
        dx = 0
        dy = 1
    if direction == 3: # south
        dx = 1
        dy = 0
    if direction == 0: # west
        dx = 0
        dy = -1
    if direction == 2: # north
        dx = -1
        dy = 0

    neighbors.append((x_coord, y_coord, dirs[direction][0]))
    neighbors.append((x_coord, y_coord, dirs[direction][1]))

    new_x_coord = x_coord + dx
    new_y_coord = y_coord + dy

    # no wall collision
    if new_x_coord > gridsize-1 or new_y_coord > gridsize-1:
        return neighbors
    
    elif new_x_coord < 0 or new_y_coord < 0:
        return neighbors
    # no block collision 
    
    elif grid[new_x_coord, new_y_coord] == WALL_CHAR:
        return neighbors
    # add neighbor 
    
    else:
        neighbors.append((new_x_coord, new_y_coord, direction))
        return neighbors


def construct_graph_from_grid(grid, node_to_idx):
    grid_size = len(grid)
    edge_list = []   
    points_of_interest = node_to_idx 
    for node ,idx in points_of_interest.items():

        nbs = get_neighbors(node, grid, gridsize=grid_size)
        edge_node_1 = idx
        for ele in nbs:
            edge_node_2 = node_to_idx[ele]
            edge_list.append((edge_node_1, edge_node_2))

    Graph = nx.DiGraph()
    Graph.add_edges_from(edge_list)

    return Graph, edge_list

def get_shortest_paths(G,start,goal,node_to_idx,idx_to_node):
    all_shortest_paths = []

    # get paths to all the orientations in the goal state   
    for direction_index in range(4): 
    
        goal_o = (goal[0], goal[1], direction_index)
        goal_node = node_to_idx[goal_o]
        start_node = node_to_idx[start]
        shortest_path = nx.shortest_path(G,source=start_node,target=goal_node)
        shortest_blocks = [idx_to_node[ele] for ele in shortest_path]
        all_shortest_paths.append(shortest_blocks)
    
    return all_shortest_paths


def check_minimal(exec_trace, grid, node_to_idx, idx_to_node):

    start_pos = exec_trace[0]
    end_pos = exec_trace[-1]

    Graph, _ = construct_graph_from_grid(grid, node_to_idx)
    shortest_paths = get_shortest_paths(Graph, start_pos, end_pos, node_to_idx, idx_to_node)
    lengths_shortest_paths = [len(path) for path in shortest_paths]

    is_minimal = len(exec_trace) <= min(lengths_shortest_paths)

    return is_minimal 


