
def calculate_edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

action_tokens = ['turn_left', 'turn_right', 'move']

conditions = ['bool_path_left', 'bool_path_right', 'bool_path_ahead']
remove_if = [
    'IFELSE', 'ELSE', 'e(', 'e)', 'i(', 'i)',
    'c( bool_path_ahead c)', 'c( bool_path_left c)', 'c( bool_path_right c)'
    ]

remove_while = ['WHILE c( bool_no_marker c)','w(', 'w)'] 

