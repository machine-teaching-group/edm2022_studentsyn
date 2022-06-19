import re
from scipy.spatial import distance
import numpy as np


CONST_DISSIM_SCORE = 1
CONST_QUALITY_SCORE = 1
CONST_DIVERSITY_SCORE = 1
# epsilon value
EPS = 1e-6


def compute_score(trace_info, input_task_data, env_type_initial, Z, tree_type, gridsz):
    score_dict = {}
    task_inputs_str, task_outputs_str, state_sequences, location_traces, hit_infos, locked_cells, agent_input_dirs, agent_input_locs, agent_output_dirs, agent_output_locs, input_markers, locked_wall_cells, symbolic_decisions = trace_info
    
    coverage_score(hit_infos[0], score_dict, tree_type)
    dissimilarity_score_per_trace(0, trace_info, input_task_data, score_dict, env_type_initial, tree_type, gridsz)
    quality_score_per_trace(0, state_sequences, location_traces, score_dict, env_type_initial, gridsz)
    diversity_score_per_trace(0, trace_info, score_dict, env_type_initial, Z, tree_type, gridsz)

    # mark repeat traces with score_diversity = 0 as invalid traces by giving a 0 score_total 
    if( score_dict["score_diversity"] < EPS ):
        score_dict["score_total"] = 0.0
    else:
        score_dict["score_total"] = ( score_dict["score_coverage"] 
                            + ( CONST_DISSIM_SCORE * score_dict["score_dissimilarity"] ) 
                            + ( CONST_QUALITY_SCORE * score_dict["score_quality"] ) 
                            + ( CONST_DISSIM_SCORE * score_dict["score_diversity"] ) ) / 4.0

    return score_dict


def diversity_score_per_trace(idx, trace_info, score_dict, env_type_initial, Z, tree_type, gridsz):
    if( len(Z) == 0 ):
        score_dict["score_diversity"] = 1
        score_dict["f_score_min_sym_diss"] = "NA"

    else:
        score_dict["f_score_sym_diss"] = []
        score_dict["score_diversity"] = min( [dissimilarity_score_per_trace_2(trace_info, score_dict, env_type_initial, z, tree_type, gridsz) for z in Z] )
        score_dict["f_score_min_sym_diss"] = min(score_dict["f_score_sym_diss"])


def dissimilarity_score_per_trace_2(trace_info, score_dict, env_type_initial, z, tree_type, gridsz):
    if( env_type_initial == "hoc" ):
        features = [(direction_agent_2, "f_score_direction_div", 0.25), (starting_quadrant_2, "f_score_location_div", 0.25),
        (cell_level_dissimilarity_2, "f_score_cell_diss_div", 0.25), (symbolic_diss_2, "f_score_symbolic_diss_div", 0.25)]
    else:
        features = [(direction_agent_2, "f_score_direction_div", 0.25), (starting_quadrant_2, "f_score_location_div", 0.25),
        (cell_level_dissimilarity_2, "f_score_cell_diss_div", 0.25), (symbolic_diss_2, "f_score_symbolic_diss_div", 0.25)]

    score = 0
    for f in features:
        f_score = f[0](trace_info, score_dict, env_type_initial, z, tree_type, gridsz)
        score += (f[2] * f_score)

    return score


def direction_agent_2(trace_info, score_dict, env_type_initial, z, tree_type, gridsz):
    s1 = trace_info[2][0]
    s2 = z[2][0]
    if( s1[0] == s2[0] ):
        return 0
    else:
        # higher score if dissimilar
        return 1


def starting_quadrant_2(trace_info, score_dict, env_type_initial, z, tree_type, gridsz):
    l1 = trace_info[3][0]
    l2 = z[3][0]

    start_y_s1 = int(l1[0].split("#")[0])
    start_x_s1 = int(l1[0].split("#")[1])
    start_y_s2 = int(l2[0].split("#")[0])
    start_x_s2 = int(l2[0].split("#")[1])

    quad_s1 = find_quadrant(start_x_s1, start_y_s1, gridsz)
    quad_s2 = find_quadrant(start_x_s2, start_y_s2, gridsz)

    if( quad_s1 == quad_s2 ):
        return 0
    else:
        # higher score if dissimilar
        return 1


def cell_level_dissimilarity_2(trace_info, score_dict, env_type_initial, z, tree_type, gridsz):
    if( env_type_initial == "hoc" ):
        # computing new_task_pregrid
        locked_cell = trace_info[5][0]
        task_output_str = trace_info[1][0]
        agent_output_loc = trace_info[9][0]
        all_cells = set()
        for i in range(gridsz):
            for j in range(gridsz):
                all_cells.add(str(i) + '#' + str(j))
        surrounding_wall_cells = all_cells.difference(locked_cell)
        for pos in surrounding_wall_cells:
            i = int(pos.split("#")[0])
            j = int(pos.split("#")[1])
            task_output_str[i][j] = "#" 
        x = agent_output_loc[0] 
        y = agent_output_loc[1]
        task_output_str[y][x] = "x"
        new_task_grid = task_output_str

        # computing z_task_pregrid
        locked_cell = z[5][0]
        task_output_str = z[1][0]
        agent_output_loc = z[9][0]
        all_cells = set()
        for i in range(gridsz):
            for j in range(gridsz):
                all_cells.add(str(i) + '#' + str(j))
        surrounding_wall_cells = all_cells.difference(locked_cell)
        for pos in surrounding_wall_cells:
            i = int(pos.split("#")[0])
            j = int(pos.split("#")[1])
            task_output_str[i][j] = "#" 
        x = agent_output_loc[0] 
        y = agent_output_loc[1]
        task_output_str[y][x] = "x"
        z_task_grid = task_output_str

        z_task_grid = [item for sublist in z_task_grid for item in sublist]
        new_task_grid = [item for sublist in new_task_grid for item in sublist]
        #input_pregrid_mat = "".join( ["".join(l) for l in input_task_pregrid] )
        hamming_distance = distance.hamming(new_task_grid, z_task_grid)

        # hamming distance already contains the n^2 denominator
        return norm_score(2*hamming_distance, 1)

    else:
        # work with max(pregrid hamming distance, postgrid hamming distance) in Karel
        
        # pregrid

        # computing new_task_pregrid
        task_input_str = trace_info[0][0]
        input_marker = trace_info[10][0]
        locked_wall_cell = trace_info[11][0]
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
        new_task_pregrid = task_input_str

        # computing z_task_pregrid
        task_input_str = z[0][0]
        input_marker = z[10][0]
        locked_wall_cell = z[11][0]
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
        z_task_pregrid = task_input_str
        
        z_task_pregrid = [item for sublist in z_task_pregrid for item in sublist]
        new_task_pregrid = [item for sublist in new_task_pregrid for item in sublist]
        hamming_distance_pregrid = distance.hamming(new_task_pregrid, z_task_pregrid)

        # postgrid

        new_task_postgrid = trace_info[1][0]
        z_task_postgrid = z[1][0]

        z_task_postgrid = [item for sublist in z_task_postgrid for item in sublist]
        new_task_postgrid = [item for sublist in new_task_postgrid for item in sublist]
        hamming_distance_postgrid = distance.hamming(new_task_postgrid, z_task_postgrid)

        # distance already contains the n^2 denominator
        # use min for karel-E = type-5 tree to avoid bunching of markers
        if( tree_type != "type-5" ):
            max_distance = max(hamming_distance_pregrid, hamming_distance_postgrid)
            return norm_score(2*max_distance, 1)
        else:
            min_distance = min(hamming_distance_pregrid, hamming_distance_postgrid)
            return norm_score(2*min_distance, 1)


def translate_symb_dec_tree_type_4(symb_dec):
    # replace in b->c->a order
    symb_dec = re.sub("ifel0 ifel1", "ifb", symb_dec)
    symb_dec = re.sub("ifel0 ifel0", "ifc", symb_dec)
    symb_dec = re.sub("ifel1", "ifa", symb_dec)

    return symb_dec


def symbolic_diss_2(trace_info, score_dict, env_type_initial, z, tree_type, gridsz):
    symb_dec_new_task = trace_info[12][0]
    symb_dec_z_task = z[12][0]

    # fix alignment issue in tree type 4 - ifa, ifb, ifc
    if( tree_type == "type-4" ):
        symb_dec_new_task_str = " ".join(symb_dec_new_task)
        symb_dec_z_task_str = " ".join(symb_dec_z_task)
        symb_dec_new_task_str = translate_symb_dec_tree_type_4(symb_dec_new_task_str)
        symb_dec_z_task_str = translate_symb_dec_tree_type_4(symb_dec_z_task_str)
        symb_dec_new_task = symb_dec_new_task_str.split(" ")
        symb_dec_z_task = symb_dec_z_task_str.split(" ")

    prefix_length = min( len(symb_dec_new_task), len(symb_dec_z_task) )

    hamming_distance = sum(sym1 != sym2 for sym1, sym2 in zip(symb_dec_new_task[:prefix_length], symb_dec_z_task[:prefix_length]))
    symbolic_diss_2_score = norm_score(hamming_distance, 4*get_good_val(gridsz))
    score_dict["f_score_sym_diss"].append(symbolic_diss_2_score)

    return symbolic_diss_2_score


def coverage_score(hit_info, score_dict, tree_type):
    # remove last if_false from the coverage dictionary since if is located outside while
    # we will only remove if_false, and not if_true, since we still need the if_true to hold, 
    # else the code is not minimal for the new task
    if( tree_type == "type-7" ):
        assert "if_true3" in hit_info, "if_true3 key missing from coverage dict"
        assert "if_false4" in hit_info, "if_false4 key missing from coverage dict"
        assert hit_info["if_true3"] + hit_info["if_false4"] == 1, "last if coverage not adding to 1"
        del hit_info["if_false4"]

    hits = 0
    for k, v in hit_info.items():
        # if trace hits conditional 
        if( v > 0 ):
            hits += 1

    missed_hits = len(hit_info)-hits
    if( len(hit_info) == 0 ):
        score_dict["score_coverage"] = 1
        score_dict["f_val_coverage"] = "NA"
    else:
        #score_dict["score_coverage"] = (0.5)**missed_hits
        score_dict["score_coverage"] = 1 if missed_hits == 0 else 0
        score_dict["f_val_coverage"] = hits/len(hit_info)
        

def dissimilarity_score_per_trace(idx, trace_info, input_task, score_dict, env_type_initial, tree_type, gridsz):
    if( env_type_initial == "hoc" ):
        features = [ (direction_agent, "f_score_direction", 0.333), (starting_quadrant, "f_score_location", 0.333),
        (cell_level_dissimilarity, "f_score_cell_diss", 0.333) ]
    else:
        features = [ (direction_agent, "f_score_direction", 0.333), (starting_quadrant, "f_score_location", 0.333),
        (cell_level_dissimilarity, "f_score_cell_diss", 0.333) ]

    score = 0
    for f in features:
        f_score = f[0](trace_info, input_task, env_type_initial, tree_type, gridsz)
        score += (f[2] * f_score)
        score_dict[f[1]] = f_score

    score_dict["score_dissimilarity"] = score


def quality_score_per_trace(idx, state_sequences, location_traces, score_dict, env_type_initial, gridsz):
    if( env_type_initial == "hoc" ):
        features = [ (count_move_forwards, "f_score_moves", 0.25), (count_turns, "f_score_turns", 0.25),
        (count_short_segments, "f_score_short_segments", 0.25), (count_long_segments, "f_score_long_segments", 0.25) ]
    else:
        features = [ (count_move_forwards, "f_score_moves", 0.1875), (count_turns, "f_score_turns", 0.1875),
        (count_short_segments, "f_score_short_segments", 0.1875), (count_long_segments, "f_score_long_segments", 0.1875),
        (count_put_marker, "f_score_putmarkers", 0.125), (count_pick_marker, "f_score_pickmarkers", 0.125) ]   

    score = 0
    for f in features:
        f_score = f[0](state_sequences[idx], location_traces[idx], score_dict, gridsz)
        score += (f[2] * f_score)
        score_dict[f[1]] = f_score

    score_dict["score_quality"] = score


def cell_level_dissimilarity(trace_info, input_task, env_type_initial, tree_type, gridsz):

    if( env_type_initial == "hoc" ):
        input_task_grid = input_task["pregrid"]

        # computing new_task_pregrid
        locked_cell = trace_info[5][0]
        task_output_str = trace_info[1][0]
        agent_output_loc = trace_info[9][0]

        all_cells = set()
        for i in range(gridsz):
            for j in range(gridsz):
                all_cells.add(str(i) + '#' + str(j))
        surrounding_wall_cells = all_cells.difference(locked_cell)
        for pos in surrounding_wall_cells:
            i = int(pos.split("#")[0])
            j = int(pos.split("#")[1])
            task_output_str[i][j] = "#" 
        x = agent_output_loc[0] 
        y = agent_output_loc[1]
        task_output_str[y][x] = "x"
        new_task_grid = task_output_str

        #print("before padding")
        #print(input_task_grid)
        #print(new_task_grid)

        # pad if gridsizes don't match
        input_task_grid = [[item for item in sublist] for sublist in input_task_grid]
        input_task_grid = np.array(input_task_grid)
        if( gridsz == 8 ):
            new_task_grid = np.pad(new_task_grid, 2, 'constant', constant_values="#")
        elif( gridsz == 10 ):
            new_task_grid = np.pad(new_task_grid, 1, 'constant', constant_values="#")
        elif( gridsz == 14 ):
            input_task_grid = np.pad(input_task_grid, 1, 'constant', constant_values="#")

        #print("after padding")
        #print(input_task_grid)
        #print(new_task_grid)

        # flattening grids for hamming distance
        input_task_grid = input_task_grid.flatten()
        new_task_grid = new_task_grid.flatten()
        #input_task_grid = [item for sublist in input_task_grid for item in sublist]
        #new_task_grid = [item for sublist in new_task_grid for item in sublist]
        hamming_distance = distance.hamming(input_task_grid, new_task_grid)

        # hamming distance already contains the n^2 denominator
        return norm_score(2*hamming_distance, 1)

    else:
        # work with max(pregrid hamming distance, postgrid hamming distance) in Karel

        # pregrid

        # input task pregrid
        input_task_pregrid = input_task["pregrid"]
        # computing new_task_pregrid
        task_input_str = trace_info[0][0]
        input_marker = trace_info[10][0]
        locked_wall_cell = trace_info[11][0]
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
        new_task_pregrid = task_input_str

        # pad if gridsizes don't match
        input_task_pregrid = [[item for item in sublist] for sublist in input_task_pregrid]
        input_task_pregrid = np.array(input_task_pregrid)
        if( gridsz == 8 ):
            new_task_pregrid = np.pad(new_task_pregrid, 2, 'constant', constant_values="#")
        elif( gridsz == 10 ):
            new_task_pregrid = np.pad(new_task_pregrid, 1, 'constant', constant_values="#")
        elif( gridsz == 14 ):
            input_task_pregrid = np.pad(input_task_pregrid, 1, 'constant', constant_values="#")

        input_task_pregrid = input_task_pregrid.flatten()
        new_task_pregrid = new_task_pregrid.flatten()
        #input_task_pregrid = [item for sublist in input_task_pregrid for item in sublist]
        #new_task_pregrid = [item for sublist in new_task_pregrid for item in sublist]
        hamming_distance_pregrid = distance.hamming(input_task_pregrid, new_task_pregrid)

        
        # postgrid

        # input task postgrid
        input_task_postgrid = input_task["postgrid"]
        # computing new_task_postgrid
        new_task_postgrid = trace_info[1][0]

        # pad if gridsizes don't match
        input_task_postgrid = [[item for item in sublist] for sublist in input_task_postgrid]
        input_task_postgrid = np.array(input_task_postgrid)
        if( gridsz == 8 ):
            new_task_postgrid = np.pad(new_task_postgrid, 2, 'constant', constant_values="#")
        elif( gridsz == 10 ):
            new_task_postgrid = np.pad(new_task_postgrid, 1, 'constant', constant_values="#")
        elif( gridsz == 14 ):
            input_task_postgrid = np.pad(input_task_postgrid, 1, 'constant', constant_values="#")

        input_task_postgrid = input_task_postgrid.flatten()
        new_task_postgrid = new_task_postgrid.flatten()

        input_task_postgrid = [item for sublist in input_task_postgrid for item in sublist]
        new_task_postgrid = [item for sublist in new_task_postgrid for item in sublist]
        hamming_distance_postgrid = distance.hamming(input_task_postgrid, new_task_postgrid)

        # distance already contains the n^2 denominator
        if( tree_type != "type-5" ):
            max_distance = max(hamming_distance_pregrid, hamming_distance_postgrid)
            return norm_score(2*max_distance, 1)
        else:
            min_distance = min(hamming_distance_pregrid, hamming_distance_postgrid)
            return norm_score(2*min_distance, 1)


def direction_agent(trace_info, input_task, env_type_initial, tree_type, gridsz):
    s1 = trace_info[2][0]
    s2 = input_task["agent_input_dir"]
    if( s1[0] == s2 ):
        return 0
    else:
        # higher score if dissimilar
        return 1


def find_quadrant(x, y, gridsz):
    if( (x<=gridsz/2) and (y<=gridsz/2) ):
        quad = "top_left"
    elif( (x>gridsz/2) and (y<=gridsz/2) ):
        quad = "top_right"
    elif( (x<=gridsz/2) and (y>gridsz/2) ):
        quad = "bottom_left"
    else:
        quad = "bottom_right"
    # ordering to check quadrants is important,
    # first check four corners then check centre
    if( gridsz == 8 ):
        if( (x>=3) and (x<=4) and (y>=3) and (y<=4) ):
            quad = "centre"        
    elif( gridsz == 12 ):
        if( (x>=4) and (x<=7) and (y>=4) and (y<=7) ):
            quad = "centre"
    elif( gridsz == 14 ):
        if( (x>=5) and (x<=9) and (y>=5) and (y<=9) ):
            quad = "centre"        
    elif( gridsz == 16 ):
        if( (x>=5) and (x<=11) and (y>=5) and (y<=11) ):
            quad = "centre"  

    return quad


def starting_quadrant(trace_info, input_task, env_type_initial, tree_type, gridsz):
    l1 = trace_info[3][0]
    l2 = input_task["agent_input_loc"]

    start_y_s1 = int(l1[0].split("#")[0])
    start_x_s1 = int(l1[0].split("#")[1])

    quad_s1 = find_quadrant(start_x_s1, start_y_s1, gridsz)
    quad_s2 = find_quadrant(l2[0], l2[1], input_task["input_gridsz"])

    if( quad_s1 == quad_s2 ):
        return 0
    else:
        # higher score if dissimilar
        return 1


def to_action_string(state_seq):
    action_str = ""
    for i in range(len(state_seq)):
        if( state_seq[i] == "move" ):
            action_str += "M"
        elif( state_seq[i] == "turn_left" ):
            action_str += "L"
        elif( state_seq[i] == "turn_right" ):
            action_str += "R"
        elif( state_seq[i] == "pick_marker" ):
            action_str += "I"
        elif( state_seq[i] == "put_marker" ):
            action_str += "U"
        else:
            continue
    return action_str


def count_put_marker(s1, l1, score_dict, gridsz):
    s1_put_marker_counts = s1.count("put_marker")
    score_dict["f_val_putmarkers"] = s1_put_marker_counts
    
    return norm_score(s1_put_marker_counts, get_good_val(gridsz))


def count_pick_marker(s1, l1, score_dict, gridsz):
    s1_pick_marker_counts = s1.count("pick_marker")
    score_dict["f_val_pickmarkers"] = s1_pick_marker_counts
    
    return norm_score(s1_pick_marker_counts, get_good_val(gridsz))


def count_short_segments(s1, l1, score_dict, gridsz):
    action_str = to_action_string(s1)
    short_segments = re.findall(r'MMM+', action_str)
    count_short_segments = len(short_segments)
    score_dict["f_val_short_segments"] = count_short_segments

    return norm_score(count_short_segments, 5)


def count_long_segments(s1, l1, score_dict, gridsz):
    action_str = to_action_string(s1)
    long_segments = re.findall(r'MMMMM+', action_str)
    count_long_segments = len(long_segments)
    score_dict["f_val_long_segments"] = count_long_segments

    return norm_score(count_long_segments, 3) 


def count_move_forwards(s1, l1, score_dict, gridsz):
    s1_moves = s1.count("move")
    score_dict["f_val_moves"] = s1_moves

    return norm_score(s1_moves, ( 2*get_good_val(gridsz) ))


def count_turns(s1, l1, score_dict, gridsz):
    s1_turns = s1.count("turn_left") + s1.count("turn_right")
    score_dict["f_val_turns"] = s1_turns

    return norm_score(s1_turns, get_good_val(gridsz))


def norm_score(val, good_val):
    return min(1, val/good_val)


def diff(x, y, thresh, norm):
    if( (x-y <= thresh) and (x-y >= 0) ):
        return 1
    else:
        return ( 1 - min(1, abs(x-y)/float(norm)) )


def get_good_val(gridsz):
    max_dim = gridsz

    # return effective dimension area
    return max_dim-2

