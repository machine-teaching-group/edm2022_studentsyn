class ReVisit(Exception):
    "raised when visiting the same state twice"
    pass 

class WallCrash(Exception):
    "raised when attemptint to move into a blocked cell"
    pass 

class LoopEnd(Exception):
    pass 

def get_num_blocks(token_list):
    TOKENS = {
        'WHILE', 'REPEAT', 'IF', 
        'IFELSE', 'move', 
        'turn_right', 'turn_left'
    }
    num_blocks = 0
    for token in token_list:
        if token in TOKENS :
            num_blocks += 1

    return num_blocks

AGENT_WEST = 0
AGENT_EAST = 1
AGENT_NORTH = 2
AGENT_SOUTH = 3
WALL = 4
GOAL = 5

DIR_CHAR_TO_INT = {
    "<" : 0, 
    ">" : 1 , 
    "^" : 2 , 
    "v" : 3
}

'''
agent        compass     index 
------------------------------
(0,-1)      north       2 
(0,1)       south       3
(-1,0)      west        0
(1,0)       east        1 

coordinates are written in (y,x):
    y axis is vertical 
    x axis is horizontal 

state rep : 
    0: agent facing West
    1: agent facing East
    2: agent facing North
    3: agent facing South
    4: wall
    5: goal 
'''

hoc4_tokens = ['DEF','run','move','m(','m)','turn_left','turn_right']
hoc18_tokens = ['DEF','run','move','m(','m)','turn_left','turn_right',
                'IFELSE','ELSE','e(','e)','i(','i)','WHILE','w(','w)',
                'c(','c)','bool_no_marker','bool_path_left','bool_path_right',
                'bool_path_ahead']


