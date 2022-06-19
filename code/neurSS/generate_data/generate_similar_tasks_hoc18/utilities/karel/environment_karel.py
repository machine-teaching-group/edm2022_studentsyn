'''
code based on https://github.com/carpedm20/karel, 
and https://github.com/alts/karel
'''

from collections import Counter
import numpy as np

from ..baseclasses.environment_base import border_mask, world_condition, Environment
from .agent_karel import _AgentKarel
from .task_karel import _TaskKarel
from ..utils import WallCrashError, \
                ExecutionTimeoutError, PickEmptyMarkerError, PutMaxMarkerError, \
                ExceededPreFlippedCoinsError, CoinFlipMismatchError


# define decorator function
marker_action = world_condition


class EnvironmentKarel(Environment):
    # class attributes
    # '+' in an hoc task represents a goal, 'x' represents a marker in a karel task
    # both are treated as markers in internal code
    MARKER_CHARS = ['x', '+']
    
    # starting_coordinates for different task grid sizes
    starting_coordinates = {
        8 : {
            1 : (1, 6),
            2 : (1, 1),
            3 : (6, 1),
            4 : (6, 6),
            5 : (4, 3)
        },
        10 : {
            1 : (1, 8),
            2 : (1, 1),
            3 : (8, 1),
            4 : (8, 8),
            5 : (5, 4)
        },
        12 : {
            1 : (2, 9),
            2 : (2, 2),
            3 : (9, 2),
            4 : (9, 9),
            5 : (6, 5)
        },
        14 : {
            1 : (3, 10),
            2 : (3, 3),
            3 : (10, 3),
            4 : (10, 10),
            5 : (7, 6)
        },
        16 : {
            1 : (4, 11),
            2 : (4, 4),
            3 : (11, 4),
            4 : (11, 11),
            5 : (8, 7)
        }
    }
    
    def __init__( 
            self, state=None, world_size=(12,12), 
            world_path=None, mode="synthesis", rng=None, wall_ratio=0.1, debug=False,
            marker_ratio=0.1, max_marker_in_cell=1, state_sequence=[], 
            prob_front_is_clear=0.5, prob_left_is_clear=0.5, prob_right_is_clear=0.5,
            prob_markers_present=0.3, prob_no_markers_present=0.7,
            max_func_calls=10000, parse_mode="compact", allow_backtracking = False, 
            coin_flips=None, input_task_constraints_file_path=None):
        super().__init__( rng, debug )

        self.max_marker = max_marker_in_cell
        self.allow_backtracking = allow_backtracking
        # use mcts pre flipped coins if available
        self.coin_flips = coin_flips
        self.coin_flip_idx = 0

        if (mode == "inverse"):
            self.locked_empty_cells = set()
            self.locked_wall_cells = set()
            self.locked_marker_cells = set()
            self.input_marker_cells = set()

            self.prob_front_is_clear = prob_front_is_clear
            self.prob_left_is_clear = prob_left_is_clear
            self.prob_right_is_clear = prob_right_is_clear
            self.prob_markers_present = prob_markers_present
            self.prob_no_markers_present = prob_no_markers_present

            self.inverse_world(world_size, input_task_constraints_file_path)
            self.mode = "inverse"

        else:
            self.mode = "synthesis"
            if state is not None:
                self.parse_state(state)
            elif world_path is not None:
                self.parse_world(world_path, parse_mode)
            elif world_size is not None:
                self.random_world(world_size, max_marker_in_cell, wall_ratio, marker_ratio)
            else:
                raise Exception(" [!] one of `world_size`, `world_path` and `world` should be passed")


    # virtual method overriding
    def inverse_world(self, world_size, input_task_constraints_file_path):
        # gridz = height = width in our case since we deal with square grids
        height, width = world_size
        
        self.width = width
        self.height = height

        if height <= 2 or width <= 2:
            raise Exception(" [!] `height` and `width` should be larger than 2")       

        # blank world
        world = np.chararray((height, width))
        world[:] = "." 
        markers = []

        # border walls all around world
        self.border_mask_inverse(world, self.WALL_CHAR)

        # read input task constraints file
        if( input_task_constraints_file_path != None ):
            lines = []
            flag = False
            with open(input_task_constraints_file_path) as f:
                for line in f.readlines():
                    if( "agentloc" in line ):
                        break
                    elif( "agentdir" in line ):
                        break
                    elif( flag == True ):
                        line = "".join(line.strip().split("\t")[1:])
                        lines.append(line)
                    elif("pregrid" in line):
                        flag = True
            for y, line in enumerate(lines):
                #print(line)
                for x, char in enumerate(line.strip()):
                    if char in self.MARKER_CHARS:
                        markers.append((x, y))
                        world[y, x] = '.'
                        self.locked_marker_cells.add( str(y) + '#' + str(x) )
                        self.input_marker_cells.add( str(y) + '#' + str(x) )
                    elif char == self.WALL_CHAR :
                        world[y, x] = '#'
                        self.locked_wall_cells.add( str(y) + '#' + str(x) )
                    elif char == self.EMPTY_CHAR :
                        world[y, x] = '.'
                        self.locked_empty_cells.add( str(y) + '#' + str(x) )
                    elif char == self.UNKNOWN_CHAR :
                        pass
                    else:
                        raise Exception(" [!] `{}` is not a valid character".format(char))


        if( self.coin_flips != None ):
            if( self.coin_flip_idx < len(self.coin_flips) ):
                val = self.coin_flips[self.coin_flip_idx]
                self.coin_flip_idx += 1
            else:
                raise ExceededPreFlippedCoinsError
            
            if( "fixed" in val ):
                val = val.split("-")
                x = int(val[1]) - 1
                y = int(val[2]) - 1
                
                direction = val[3]
                if( direction == "north" ):
                    direction = (0, -1)
                elif( direction == "east" ):
                    direction = (1, 0)
                elif( direction == "south" ):
                    direction = (0, 1)
                else:
                    direction = (-1, 0)
            
            else:
                quadrant = int(val[0])
                # gridz = height = width in our case since we deal with square grids
                x, y = self.starting_coordinates[height][quadrant]

                direction = val[1]
                if( direction == "N" ):
                    direction = (0, -1)
                elif( direction == "E" ):
                    direction = (1, 0)
                elif( direction == "S" ):
                    direction = (0, 1)
                else:
                    direction = (-1, 0)

            agent = _AgentKarel((x, y), direction)
            # x in 2D real world corresponds to column index in list of lists, therefore index swap
            world[y, x] = '.'
            self.agent_input_dir = direction
            self.agent_input_loc = (x,y)
            self.state_sequence.append(direction)

        else:
            # create AgentHoc
            starting_coordinates = [[2,2], [2,9], [9,2], [9,9], [6,5]]
            starting_coordinates_idx = self.rng.randint(5)
            x, y = starting_coordinates[starting_coordinates_idx]
            direction = self.rng.randint(4)
            agent = _AgentKarel((x, y), ((-1, 0), (1, 0), (0, -1), (0, 1))[direction])
            # x in 2D real world corresponds to column index in list of lists, therefore index swap
            world[y, x] = '.'
            self.agent_input_dir = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
            self.agent_input_loc = (x,y)
            self.state_sequence.append(((-1, 0), (1, 0), (0, -1), (0, 1))[direction])

        agent_loc = str(y) + '#' + str(x)
        self.locked_empty_cells.add(agent_loc)
        self.location_trace.append(agent_loc)

        # check agentloc cell is not a locked_wall_cell
        if( agent_loc in self.locked_wall_cells ):
            #self.locked_wall_cells.remove(agent_loc)
            raise WallCrashError

        world = world.astype(str).tolist()
        # store data of environment in task object
        self.task = _TaskKarel(world, agent, self.agent_direction, markers, self.max_marker)


    # virtual method overriding
    def random_world(self, world_size, max_marker_in_cell, wall_ratio, marker_ratio):
        height, width = world_size
        self.width = width
        self.height = height

        if height <= 2 or width <= 2:
            raise Exception(" [!] `height` and `width` should be larger than 2")

        # blank world
        world = np.chararray((height, width))
        world[:] = "."

        # wall
        wall_array = self.rng.rand(height, width)
        world[wall_array < wall_ratio] = self.WALL_CHAR
        border_mask(world, self.WALL_CHAR)

        # create AgentKarel
        x, y, direction = self.rng.randint(1, width-1), \
                self.rng.randint(1, height-1), self.rng.randint(4)
        agent = _AgentKarel((x, y), ((-1, 0), (1, 0), (0, -1), (0, 1))[direction])
        world[y, x] = '.'

        # markers
        marker_array = self.rng.rand(height, width)
        marker_array = (wall_array >= wall_ratio) & (marker_array < marker_ratio)
        border_mask(marker_array, False)

        markers = []
        for (y, x) in zip(*np.where(marker_array > 0)):
            markers.append((x, y))

        world = world.astype(str).tolist()
        # store data of environment in task object
        self.task = _TaskKarel(world, agent, self.agent_direction, markers, self.max_marker)


    def parse_world(self, world_path, parse_mode):
        directions = {
            '>': (1, 0),
            '^': (0, -1),
            '<': (-1, 0),
            'v': (0, 1),
        }

        if( parse_mode == "tab_separated" ):
            lines = []
            with open(world_path) as f:
                flag = False
                for line in f.readlines():
                    if("agentloc" in line):
                        pos = line.strip().split("\t")[1]
                        x = int(pos.split(",")[0][1:]) - 1
                        y = int(pos.split(",")[1][:-1]) - 1
                        agent_loc = str(y+1) + '#' + str(x+1)
                        self.location_trace.append(agent_loc)
                        flag = False
                    if("agentdir" in line):
                        if( "east" in line ):
                            char = ">"
                        elif( "west" in line ):
                            char = "<"
                        elif( "north" in line ):
                            char = "^"
                        else:
                            char = "v"
                        break
                    if( flag == True ):
                        line = "".join(line.strip().split("\t")[1:])
                        lines.append(line)
                    if("pregrid" in line):
                        flag = True
            agent = _AgentKarel((x + 1, y + 1), directions[char])
            self.state_sequence.append(directions[char])
        else:
            with open(world_path) as f:
                lines = f.readlines()

        world = [[]]
        markers = []
        for y, line in enumerate(lines):
            row = []
            for x, char in enumerate(line.strip()):
                if char in self.AGENT_CHARS:
                    agent = _AgentKarel((x + 1, y + 1), directions[char])
                    agent_loc = str(y+1) + '#' + str(x+1)
                    self.state_sequence.append(directions[char])
                    self.location_trace.append(agent_loc)
                    char = '.'
                elif char in self.MARKER_CHARS:
                    markers.append((x + 1, y + 1))
                    char = '.'
                elif char in [self.WALL_CHAR, self.EMPTY_CHAR]:
                    pass
                else:
                    raise Exception(" [!] `{}` is not a valid character".format(char))
                row.append(char)
            world.append([self.WALL_CHAR] + row + [self.WALL_CHAR])

        world.append([])
        for _ in range(len(world[1])):
            world[0].append(self.WALL_CHAR)
            world[-1].append(self.WALL_CHAR)
        self.width = len(world[0])
        self.height = len(world)
        # store data of environment in task object
        self.task = _TaskKarel(world, agent, self.agent_direction, markers, self.max_marker)


    # virtual method overriding
    def parse_state(self, state):
        height, width, _ = state.shape

        world = np.chararray((height, width))
        world[:] = "."

        # wall
        world[state[:,:,4] == 1] = self.WALL_CHAR

        # agent
        y, x, facing_idx = zip(*np.where(state[:,:,:4]==1)).__next__()
        agent = _AgentKarel((x, y), ((0, -1), (0, 1), (-1, 0), (1, 0))[facing_idx])

        # markers
        markers = []
        max_marker = len(state[0,0,6:])
        for num in range(1, max_marker + 1):
            for (y, x) in zip(*np.where(state[:,:,5+num] == 1)):
                for _ in range(num):
                    markers.append((x, y))

        world = world.astype(str).tolist()
        self.width = len(world[0])
        self.height = len(world)
        # store data of environment in task object
        self.task = _TaskKarel(world, agent, self.agent_direction, markers, self.max_marker)


    # virtual method overriding
    def draw(self, prefix="", skip_number=False, with_color=False, no_print=False):
        canvas = np.array(self.task.world)

        
        for (x, y), count in Counter(self.task.markers).items():
            canvas[y][x] = str(count)
        
        '''
        for (x, y), count in Counter(self.task.markers).items():
            canvas[y][x] = "+"

        return canvas
        '''
        
        canvas[self.task.agent.position[1]][self.task.agent.position[0]] = self.agent_char()

        texts = []
        for idx, row in enumerate(canvas):
            row_text = "".join(row)
            if skip_number:
                row_text = re.sub('\d', self.MARKER_CHAR, row_text)

            if idx == 0:
                if with_color:
                    text = "{}{}{}{}".format(
                            Tcolors.OKBLUE, prefix, Tcolors.ENDC, row_text)
                else:
                    text = "{}{}".format(prefix, row_text)
            else:
                text = "{}{}".format(len(prefix) * " ", row_text)

            if with_color:
                text = re.sub('.'.format(self.WALL_CHAR), lambda x: self._color_fn(x), text)

            if not no_print:
                print(text)
            texts.append(text)

        if no_print:
            return texts


    # virtual method overriding
    def draw_array(self, prefix="", skip_number=False, with_color=False, no_print=False):
        canvas = np.array(self.task.world)
        for (x, y), count in Counter(self.task.markers).items():
            canvas[y][x] = "x"

        return canvas



    # defining marker actions unique to Karel environment
    @marker_action
    def pick_marker(self):
        '''Pick marker'''
        curr_loc = str(self.task.agent.position[1]) + '#' + str(self.task.agent.position[0])
        if( self.mode == "inverse" ):
            if( curr_loc not in self.locked_marker_cells ): 
                self.input_marker_cells.add(curr_loc)
                self.locked_marker_cells.add(curr_loc)
                self.state_sequence.append("pick_marker")  
            else:
                position = self.task.agent.position
                for i, coord in enumerate(self.task.markers):
                    if coord == self.task.agent.position:
                        del self.task.markers[i]
                        self.task.agent.pick_marker()  
                        self.state_sequence.append("pick_marker")  
                        break
                else:
                    raise PickEmptyMarkerError
                 
        else:
            position = self.task.agent.position
            for i, coord in enumerate(self.task.markers):
                if coord == self.task.agent.position:
                    del self.task.markers[i]
                    self.task.agent.pick_marker()  
                    self.state_sequence.append("pick_marker")  
                    break
            else:
                raise PickEmptyMarkerError


    @marker_action
    def put_marker(self):
        '''Put marker'''
        curr_loc = str(self.task.agent.position[1]) + '#' + str(self.task.agent.position[0])
        if( self.mode == "inverse" ):
            if( curr_loc not in self.locked_marker_cells ): 
                self.locked_marker_cells.add(curr_loc)
            if( not self.task.agent.holding_markers() ):
                #raise Exception('can\'t put marker. Agent has none')
                pass
            elif( self.task.markers.count(( self.task.agent.position )) == self.max_marker ):
                raise PutMaxMarkerError
            else:
                self.task.markers.append(self.task.agent.position)
                self.task.agent.put_marker()
                self.state_sequence.append("put_marker")
        
        else:
            if not self.task.agent.holding_markers():
                #raise Exception('can\'t put marker. Agent has none')
                pass
            elif( self.task.markers.count(( self.task.agent.position )) == self.max_marker ):
                raise PutMaxMarkerError
            else:
                self.task.markers.append(self.task.agent.position)
                self.task.agent.put_marker() 
                self.state_sequence.append("put_marker")           
        
        return Counter(self.task.markers)[self.task.agent.position]


    # defining world condition checks unique to karel environment   
    @world_condition
    def markers_present(self):
        '''Check markers present'''
        
        curr_loc = str(self.task.agent.position[1]) + '#' + str(self.task.agent.position[0])

        if( not self.record_func_call() ):
            raise ExecutionTimeoutError
        else:
            if( self.mode == "inverse" ):
                # get a coin flip
                if( self.coin_flips != None ):
                    if( self.coin_flip_idx < len(self.coin_flips) ):
                        coin_toss = self.coin_flips[self.coin_flip_idx]
                        self.coin_flip_idx += 1
                    else:
                        # this exception should logically never be thrown
                        raise ExceededPreFlippedCoinsError
                else:
                    coin_toss = np.random.choice([0, 1], p=[1-self.prob_markers_present, self.prob_markers_present])

                if( curr_loc not in ( self.locked_marker_cells | self.locked_wall_cells ) ):
                    self.locked_marker_cells.add(curr_loc)
                         
                    if( coin_toss ):
                        self.state_sequence.append("markers_present=true")
                        # add a marker
                        # agent can only add markers it has
                        self.task.markers.append(self.task.agent.position)
                        self.input_marker_cells.add(curr_loc)
                    else:
                        self.state_sequence.append("markers_present=false")
                        pass
                
                else:
                    # check if in mcts inverse mode
                    if( self.coin_flips != None ):
                        # check for coin flip mismatch
                        # mcts intended coin toss doesn't match environment locked coin toss
                        env_locked_coin_toss = self.task.agent.position in self.task.markers
                        if( coin_toss != env_locked_coin_toss ):
                            raise CoinFlipMismatchError

                    if( self.task.agent.position in self.task.markers ):
                        self.state_sequence.append("markers_present=true")
                    else:
                        self.state_sequence.append("markers_present=false")
            else:
                    if( self.task.agent.position in self.task.markers ):
                        self.state_sequence.append("markers_present=true")
                    else:
                        self.state_sequence.append("markers_present=false")            

        
        return self.task.agent.position in self.task.markers

    @world_condition
    def no_markers_present(self):
        '''Check no markers present'''
        
        curr_loc = str(self.task.agent.position[1]) + '#' + str(self.task.agent.position[0])

        if( not self.record_func_call() ):
            raise ExecutionTimeoutError
        else:
            if( self.mode == "inverse" ):
                # get a coin flip
                if( self.coin_flips != None ):
                    if( self.coin_flip_idx < len(self.coin_flips) ):
                        coin_toss = self.coin_flips[self.coin_flip_idx]
                        #print("self.coin_flip_idx: ", self.coin_flip_idx)
                        self.coin_flip_idx += 1
                    else:
                        # this exception should logically never be thrown
                        raise ExceededPreFlippedCoinsError
                else:
                    coin_toss = np.random.choice([0, 1], p=[1-self.prob_no_markers_present, self.prob_no_markers_present])

                # use coin flip to build new environment cell
                if( curr_loc not in ( self.locked_marker_cells | self.locked_wall_cells ) ):
                    self.locked_marker_cells.add(curr_loc)
                    if( coin_toss ):
                        # don't add a marker
                        self.state_sequence.append("no_markers_present=true")
                    else:
                        self.state_sequence.append("no_markers_present=false")
                        self.task.markers.append(self.task.agent.position)
                        self.input_marker_cells.add(curr_loc)
                
                # environment cell is locked
                else:
                    # check if in mcts inverse mode
                    if( self.coin_flips != None ):
                        # check for coin flip mismatch
                        # mcts intended coin toss doesn't match environment locked coin toss
                        env_locked_coin_toss = self.task.agent.position not in self.task.markers
                        if( coin_toss != env_locked_coin_toss ):
                            raise CoinFlipMismatchError

                    if( self.task.agent.position not in self.task.markers ):
                        self.state_sequence.append("no_markers_present=true")
                    else:
                        self.state_sequence.append("no_markers_present=false") 
            else:
                    if( self.task.agent.position in self.task.markers ):
                        self.state_sequence.append("markers_present=true")
                    else:
                        self.state_sequence.append("markers_present=false")           
        
        return self.task.agent.position not in self.task.markers


    @property
    def state(self):
        """
            0: Hero facing North
            1: Hero facing South
            2: Hero facing West
            3: Hero facing East
            4: Wall
            5: 0 marker
            6: 1 marker
            7: 2 marker
            8: 3 marker
            9: 4 marker
            10: 5 marker
            11: 6 marker
            12: 7 marker
            13: 8 marker
            14: 9 marker
            15: 10 marker
        """
        state = super().state()

        # 5 ~ 15: marker counter
        for (x, y), count in Counter(self.task.markers).items():
            state[y][x][5] = 0
            state[y][x][5 + count] = 1
            #state[y][x][min(5 + count, self.max_marker)] = 1

        # draw2d(state[:,:,5])
        return state


    # yacc will use these names to call above functions
    bool_marker = markers_present
    bool_no_marker = no_markers_present


    pickMarker = pick_marker
    putMarker = put_marker




