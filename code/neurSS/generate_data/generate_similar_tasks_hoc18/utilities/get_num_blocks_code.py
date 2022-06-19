TOKENS = {
    'WHILE' : 0,
    'REPEAT' : 0,
    'IF' : 0, 
    'IFELSE' : 0,
    'move' : 0, 
    'turn_right' : 0, 
    'turn_left' : 0,
    'pick_marker' : 0, 
    'put_marker' : 0
    }


def get_num_blocks(code_tokens):
    code_tokens = code_tokens.split(" ")
    num_blocks = 0
    for token in code_tokens:
        if( token in TOKENS ):
            num_blocks += 1

    return num_blocks