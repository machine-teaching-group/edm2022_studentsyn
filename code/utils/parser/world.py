import numpy as np
from code.utils.utils import *
from .parser import Parser
from .utils import WallCrash, ReVisit, LoopEnd, DIR_CHAR_TO_INT


class Agent(object):

    def __init__(self, x,y, direction):
        self.set_position(x,y,direction)
    
    def move(self):
        self.position = (
            self.position[0] + self.facing[0],
            self.position[1] + self.facing[1]
        )
    def turn_left(self):
        self.facing = (
            self.facing[1],
            -self.facing[0]
        )
    def turn_right(self):
        self.facing = (
            -self.facing[1],
            self.facing[0]
        )

    def set_position(self,x,y,direction):
        self.position = (x,y)
        self.facing = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]


class World:

    HERO_CHARS = '<^>v'
    WALL_CHAR = '#'
    EMPTY_CHAR = '.'
    GOAL_CHAR = '+'
    TRACE_CHAR = '*'
    VECTOR_SIZE = 6


    def read_from_file(self, file_path = None, lines = None):

        assert file_path is not None or lines is not None  
        if file_path is not None : 
            with open(file_path,'r') as f : 
                lines = f.readlines()

        lines_parsed = []
        for line in lines : 
            line = line.strip().split()
            if line == [] or line[1].isnumeric() : continue 
            lines_parsed.append(line[1:])

        self.grid_size = len(lines_parsed[0])
        self.world = np.array(lines_parsed)

        agent_x_pos ,agent_y_pos = np.where(np.isin(self.world , list(self.HERO_CHARS)))            
        agent_x_pos, agent_y_pos = agent_x_pos[0],agent_y_pos[0]
        
        agent_direction = DIR_CHAR_TO_INT[self.world[agent_x_pos,agent_y_pos][0]]  
        self.world[agent_x_pos,agent_y_pos] = self.EMPTY_CHAR
        self.init_pos = (agent_y_pos,agent_x_pos,agent_direction)

        goal_x_pos, goal_y_pos = np.where(self.world == self.GOAL_CHAR)        

        self.goal = goal_x_pos, goal_y_pos
        self.world[goal_x_pos,goal_y_pos] = self.EMPTY_CHAR

        self.goal_set = True 
        self.agent = Agent(*self.init_pos)
 
        self.finalize_init()


    def read_from_state(self, state):

        self.grid_size = state.shape[-1]

        self.world = np.zeros((self.grid_size,self.grid_size),'U1')
        self.world.fill(self.EMPTY_CHAR)
        self.world[np.where(state[4] == 1)] = self.WALL_CHAR


        vector_dim, pos_y, pos_x = np.where(state != 0 )            
        if pos_y.size != 0 : 
            agent_y_pos = pos_y[vector_dim <= 3][0]
            agent_x_pos = pos_x[vector_dim <= 3][0]

        dir_idx =  vector_dim[vector_dim <= 3][0]

        self.init_pos = (agent_x_pos, agent_y_pos, dir_idx)
 
        self.agent = Agent(*self.init_pos)
        self.goal = np.where(state[5])

        self.zero_state = np.zeros((self.VECTOR_SIZE , self.grid_size, self.grid_size))

        self.goal_set = True 
        self.finalize_init()
        

    def read_from_idxs(self,idxs, IMG_FEAT, IMG_SIZE):
        state =  np.zeros(IMG_FEAT)
        state[idxs] = 1
        state = state.reshape(IMG_SIZE).astype(int)      
        self.read_from_state(state)



    def create_random_empty(self, grid_size, rng = None):

        if rng == None : 
            rng = np.random.RandomState(123)
        
        self.rng = rng
        self.grid_size = grid_size            

        agent_x_pos = self.rng.randint(1, self.grid_size-1) 
        agent_y_pos = self.rng.randint(1, self.grid_size-1)
        direction = self.rng.randint(4)
        
        self.init_pos = (agent_x_pos ,agent_y_pos ,direction)
        self.agent = Agent(*self.init_pos)

        self.world = np.zeros((self.grid_size,self.grid_size),'U1')
        self.world.fill(self.EMPTY_CHAR)

        self.world[[0,self.grid_size-1],:] = self.WALL_CHAR
        self.world[:,[0,self.grid_size-1]] = self.WALL_CHAR

        self.goal_set = False 
        self.finalize_init()
    



    def finalize_init(self):

        # grid properties 
        self.IMG_FEAT = self.grid_size * self.grid_size * self.VECTOR_SIZE 
        self.IMG_SIZE = (self.VECTOR_SIZE ,self.grid_size,self.grid_size)
        self.crashed = False        
        self.trace = []
        self.track_visit = False 
        self.correct_syntax = True 
    

    def get_state(self):       
        state = np.zeros(self.IMG_SIZE)
        agent_x_pos, agent_y_pos = self.agent.position
        state[self.facing_idx(), agent_y_pos, agent_x_pos] = 1
        
        # add walls
        for jdx, row in enumerate(self.world):
            for idx, char in enumerate(row):
                if char == self.WALL_CHAR:
                    state[4][jdx][idx] = 1        

        if self.goal_set : 
            state[5][self.goal[0],self.goal[1]] = 1 
        else : 
            raise Exception('goal is unkown')

        return state 

    def get_idxs(self):

        world_out = self.get_state()
        idxs = np.where(world_out.flatten() != 0)[0]

        return idxs.tolist()

    # -------------- Run program utils ------------
        
    # get agent coordinates for facing 
    def facing_north(self):
        return self.agent.facing[1] == -1

    def facing_south(self):
        return self.agent.facing[1] == 1

    def facing_west(self):
        return self.agent.facing[0] == -1

    def facing_east(self):
        return self.agent.facing[0] == 1

    # check clear 
    def path_ahead(self):
        next_x = self.agent.position[0] + self.agent.facing[0]
        next_y = self.agent.position[1] + self.agent.facing[1]
        return self.world[next_y,next_x] not in [self.WALL_CHAR]

    def path_left(self):
        next_x = self.agent.position[0] + self.agent.facing[1]
        next_y = self.agent.position[1] - self.agent.facing[0]
        return self.world[next_y,next_x] not in [self.WALL_CHAR]

    def path_right(self):
        next_x = self.agent.position[0] - self.agent.facing[1]
        next_y = self.agent.position[1] + self.agent.facing[0]
        return self.world[next_y,next_x] not in [self.WALL_CHAR]

    def out_of_bounds(self, x, y):
        return (x <= 0 or x >= self.grid_size -1) or\
               (y <= 0 or y >= self.grid_size -1)

    # get facing direction 
    # from agent to world coordinates 
    # (0,-1) ---> 2 etc 
    def facing_idx(self):
        if self.facing_north(): # (0, -1) , dir = 2 
            return 2
        elif self.facing_south(): # (0, 1), dir = 3 
            return 3
        elif self.facing_west(): # (-1, 0), dir = 0 
            return 0
        elif self.facing_east(): # (1, 0), dir = 1 
            return 1

    def add_trace(self):
        
        x, y = self.agent.position[1],self.agent.position[0]

        # if you visited the same state in the (x,y,direction) graph then we have a loop 
        if self.track_visit:
            if self.visited[x,y,self.facing_idx()] >= 1 : raise ReVisit()

        # if you visit the same position more than X = 100 tmes we have an infinite loop 
        if self.visited[x,y,self.facing_idx()] >= 100 : raise TimeoutError
        self.visited[x,y,self.facing_idx()] += 1        
        self.trace.append((x,y,self.facing_idx()))
        

    def move(self):        
        next_x = self.agent.position[0] + self.agent.facing[0]
        next_y = self.agent.position[1] + self.agent.facing[1]
        

        if not self.path_ahead() or self.out_of_bounds(next_x ,next_y):
            raise WallCrash()
        else:
            self.agent.move()

        if self.goal_set and not self.bool_no_marker() : 
            raise LoopEnd
 
    def turn_left(self):
        self.agent.turn_left()

    def turn_right(self):
        self.agent.turn_right()
    
    def bool_no_marker(self):
        agent_x_pos, agent_y_pos = self.agent.position
        goal_x_pos, goal_y_pos = self.goal 
        return not (agent_x_pos == goal_y_pos and goal_x_pos == agent_y_pos)


    def get_hero_char(self):
        index = self.agent.facing[0] + 2*self.agent.facing[1]
        return ' >v^<'[index]




    def run(self,code, track_visit = False, debug = False):
        self.parser = Parser(debug = debug)
        self.parser.set_execution_env(self)
        self.debug = debug
        self.track_visit = track_visit
        self.looped = False 
        self.crashed = False 
        self.correct_syntax = True 
        self.visited =  np.zeros((self.grid_size,self.grid_size,4))    
        self.add_trace()
        run_parsed_program_fn = self.parser.yacc.parse(code)
        try :
            run_parsed_program_fn()
        except WallCrash: 
            self.crashed = True 
        except ReVisit:
            self.looped = True 
        except SyntaxError : 
            self.correct_syntax = False 

        except LoopEnd :
            pass 
        
        except TimeoutError: 
            pass

        if debug : self.print()

        if not self.goal_set : 
            self.goal_set = True
            self.goal = tuple(self.trace[-1][0:2])


    def step(self,code,debug = False):

        if not isinstance(code,list):
            code = [code]

        code = ' '.join(code)
        code = 'DEF run m( ' + code + ' m)'
        self.debug = debug 

        try : 
            run_parsed_program_fn = self.yacc.parse(code)
            run_parsed_program_fn()

        except WallCrash: 
            self.crashed = True 
        except ReVisit:
            self.looped = True 
        except SyntaxError : 
            self.correct_syntax = False 
        except LoopEnd :
            pass 

    def print(self, prefix="", print_fn = print):
        print_fn(self.__repr__(prefix))

    def __repr__(self,  prefix=""):
        canvas = np.array(self.world)
        for point in self.trace : 
            canvas[point[0],point[1]] = self.TRACE_CHAR  

        canvas[self.agent.position[1]][self.agent.position[0]] = self.get_hero_char()
        
        if self.goal_set:
            canvas[self.goal[0],self.goal[1]] = self.GOAL_CHAR
    
        texts = []
        for idx, row in enumerate(canvas):
            row_text = "".join(row)
            if idx == 0:
                text = "{}{}".format(prefix, row_text)
            else:
                text = "{}{}".format(len(prefix) * " ", row_text)
            texts.append(text)

        str_repr = '\n'.join(texts)
        return str_repr
    
def test():
    world_obj = World()
    world_obj.create_random_empty(12)
    world_obj.goal = (11,11)
    test_code = 'DEF run m( WHILE c( bool_no_marker c) w( turn_left move move w) m)'
    print('press enter for next step')
    world_obj.run(test_code, debug = True)

if __name__ == '__main__':
    test()