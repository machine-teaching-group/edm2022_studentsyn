import numpy as np 

from copy import deepcopy 
from code.utils.utils import * 

PROB_FRONT_IS_CLEAR = 0.5
PROB_LEFT_IS_CLEAR = 0.5
PROB_RIGHT_IS_CLEAR = 0.5
PROB_MARKERS_PRESENT = 0.5
EPSILON = 1e-6

env_type = 'hoc' 
env_type_execution = 'hoc'
DEBUG = False 

facing2dir =  {(-1,0) : "west",
                (1,0) : "east", 
                (0,-1) : "north", 
                (0,1) : "south"}

facing2idx = {(-1,0) : 0,
              (1,0)  : 1, 
              (0,-1) : 2, 
              (0,1) : 3}




def mutate_grid(grid, random_generator, grid_size, prob_rand_line = 0.6, 
    prob_scatter = 0.3, scatter_type = 'variant_1', EMPTY_CHAR = '.', WALL_CHAR = '#'):
    

    grid_mutated = deepcopy(grid)
    free_cells = np.where(grid_mutated != WALL_CHAR)
    free_cells_x_pos = free_cells[0]
    free_cells_y_pos = free_cells[1]

    generated_grids = [grid]

    # randomly add lines as `distractor` paths
    for pos_x, pos_y in zip(free_cells_x_pos, free_cells_y_pos):

        if random_generator.random() > prob_rand_line : continue 

        line_size = random_generator.randint(1 ,grid_size-1 ) // 2
        min_x = max(pos_x - line_size, 1)
        max_x = min(grid_size-2, pos_x + line_size )
        points_x = range(min_x, max_x)
        grid_mutated[points_x, pos_y] = EMPTY_CHAR

        generated_grids += [deepcopy(grid_mutated)]

        if random_generator.random() > prob_rand_line : continue 

        line_size = random_generator.randint(1 ,grid_size-1 ) // 2
        min_y = max(pos_x - line_size, 1)
        max_y = min(grid_size-2, pos_x + line_size )
        points_y = range(min_y, max_y)
        grid_mutated[pos_x, points_y] = EMPTY_CHAR

        generated_grids += [deepcopy(grid_mutated)]


    if scatter_type == "variant_1":

        num_scatter_points = random_generator.randint(0,10)   
        # randomly add empty cellss
        for _ in range(num_scatter_points):
            pos_x = random_generator.randint(1 ,grid_size-2)
            pos_y = random_generator.randint(1, grid_size-2)
            grid_mutated[pos_x, pos_y] = EMPTY_CHAR

    elif scatter_type == 'variant_2': 

        prob_scatter_array = random_generator.rand(grid_size, grid_size)
        grid_mutated[prob_scatter_array < prob_scatter] = EMPTY_CHAR

    else : 
        print('unknown scatter type')
        exit(1)

    generated_grids += [deepcopy(grid_mutated)]

    return generated_grids
