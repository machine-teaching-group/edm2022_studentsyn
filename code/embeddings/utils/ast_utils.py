
import enum

class NodeType(enum.IntEnum):
    LIST = 1
    MOVE_FORWARD = 2
    TURN = 14
    TURN_LEFT = 4
    TURN_RIGHT = 8
    FOREVER = 5
    IF_ELSE = 12
    IF_ELSE_PATH_FORWARD = 3
    IF_ELSE_PATH_LEFT = 6
    IF_ELSE_PATH_RIGHT = 9
    LIST_END = 7

    def is_IF_ELSE(self):
        return int(self) % 3 == 0


hoc4_types = ['maze_turn', 'maze_moveForward']
hoc18_types = ['maze_ifElse', 'maze_forever'] + hoc4_types
all_types = hoc18_types


node_type_parameters = {
    'maze_turn': [ 'turnLeft', 'turnRight' ],
    'maze_ifElse': [ 'isPathForward', 'isPathLeft', 'isPathRight' ],
}


def merge_node_type_parameter(node_type: str, parameter):
    ''' Merges node type and its parameter into a single label.
        E.g.: (maze_turn, turnLeft) -> maze_turn_turnLeft '''

    if parameter is None:
        return node_type
    assert(parameter in node_type_parameters.get(node_type, []))
    return node_type+'_'+parameter


str_to_node_type = {
    'list' : NodeType.LIST,
    'maze_moveForward' : NodeType.MOVE_FORWARD,
    merge_node_type_parameter('maze_turn', 'turnLeft') : NodeType.TURN_LEFT,
    merge_node_type_parameter('maze_turn', 'turnRight') : NodeType.TURN_RIGHT,
    'maze_forever': NodeType.FOREVER,
    merge_node_type_parameter('maze_ifElse', 'isPathForward'): NodeType.IF_ELSE_PATH_FORWARD,
    merge_node_type_parameter('maze_ifElse', 'isPathLeft'): NodeType.IF_ELSE_PATH_LEFT,
    merge_node_type_parameter('maze_ifElse', 'isPathRight'): NodeType.IF_ELSE_PATH_RIGHT,
}

node_type_to_str = { v: k for k, v in str_to_node_type.items() }   
        

class ASTNode:
    ''' A custom AST class that can be converted to and from JSON states.
        It is easier to perform some operations on ASTNode rather than
        directly on JSON dicts. '''

    @staticmethod
    def from_json(json_dict):
        return json_to_ast(json_dict)

    @staticmethod
    def from_int(num):
        node, r = int_to_node(num)
        assert(r == 0)
        return node

    
    def to_int(self):
        return self._hash


    def to_json(self):
        return ast_to_json(self)


    def __init__(self, node_type_str, children = [], node_type_enum=None):
        self._type_enum = node_type_enum or str_to_node_type[node_type_str]
        self._type = node_type_str
        self._children = children
        self._size = 0
        self._depth = 0
        self._hash = int(self._type_enum)
        self._n_if_else = 0
        self._n_forever = 0

        for child in children:
            self._size += child._size
            self._depth = max(self._depth, child._depth)
            self._hash = join_ints(child._hash, self._hash)
            self._n_if_else += child._n_if_else
            self._n_forever += child._n_forever
        if self._type_enum == NodeType.LIST:
            self._hash = join_ints(int(NodeType.LIST_END), self._hash)
        else:
            self._size += 1
            self._depth += 1
            if self._type_enum == NodeType.FOREVER:
                self._n_forever += 1
            elif self._type_enum.is_IF_ELSE():
                self._n_if_else += 1
                
    def size(self):
        return self._size

    def depth(self):
        return self._depth

    def children(self):
        return self._children

    def n_children(self):
        return len(self._children)

    def label(self):
        return self._type

    def label_enum(self):
        return self._type_enum

    def with_label(self, label):
        return ASTNode(label, self.children())

    def with_children(self, children):
        return ASTNode(self._type, children, self._type_enum)

    def with_ith_child(self, i, child):
        if must_be_last_node(child) and i + 1 != len(self._children) and i != -1:
            return None
        new_children = self._children[:i] + [ child ]
        if i != -1:
            new_children += self._children[i+1:]
        return ASTNode(self._type, new_children, self._type_enum)

    def with_inserted_child(self, i, child):
        if i > 0 and must_be_last_node(self.children()[i-1]):
            return None
        if must_be_last_node(child) and i != len(self._children):
            return None
        new_children = self.children().copy()
        new_children.insert(i, child)
        return ASTNode(self._type, new_children, self._type_enum)

    def with_inserted_children(self, i, children):
        old_children = self.children()
        if i > 0 and must_be_last_node(old_children[i-1]):
            return None
        if must_be_last_node(children[-1]) and i != len(self._children):
            return None
        new_children = old_children[:i] + children + old_children[i:]
        return ASTNode(self._type, new_children, self._type_enum)

    def with_removed_child(self, i):
        old_children = self.children()
        new_children = old_children[:i] + old_children[i+1:]
        return ASTNode(self._type, new_children, self._type_enum)

    def info_label_counts(self, acc=None):
        if acc is None:
            acc = dict()
            for node_type in all_types:
                for parameter in node_type_parameters.get(node_type, [None]):
                    acc[merge_node_type_parameter(node_type, parameter)] = 0
            acc['list'] = 0

        acc[self.label()] += 1
        for child in self.children():
            child.info_label_counts(acc)

        return acc

    def tokenize(self, acc=None):
        if acc is None:
            acc = []
        acc.append(int(self.label_enum()))
        for child in self.children():
            acc = child.tokenize(acc)

        # add one extra for node_type list 
        if self.label_enum() == NodeType.LIST:
            acc.append(int(NodeType.LIST_END))
        return acc
                
    def __repr__(self, offset=''):
        cs = offset + self.label() + '\n'
        for child in self.children():
            cs += offset + child.__repr__(offset + '   ')
        return cs

    def __eq__(self, other):
        return self._hash == other._hash
       # return self.label() == other.label() and self.children() == other.children()

    def __hash__(self):
        return self._hash


def must_be_last_node(node: ASTNode):
    return node._type_enum == NodeType.FOREVER


def node_types_for_dataset(dataset_id: str):
    ''' Generates all available AST node types for a dataset. '''
    if dataset_id in ['4', 'hoc4']:
        return hoc4_types
    elif dataset_id in ['18', 'hoc18']:
        return hoc18_types
    elif dataset_id == 'all':
        return all_types


def split_node_type_parameter(node_type_with_parameter: str):
    ''' Merges node-type-parameter label into node type and parameter.
        E.g.: maze_turn_turnLeft -> (maze_turn, turnLeft). '''

    splitted = node_type_with_parameter.split('_')
    parameter = splitted[-1]
    node_type = ('_').join(splitted[:-1])
    if node_type in node_type_parameters.keys():
        #assert(parameter in node_type_parameters[node_type])
        return (node_type, parameter)
    else:
        return (node_type_with_parameter, None)


def parameters_for_node_type(node_type: str):
    ''' Returns all possible parameters for a node type. '''

    return node_type_parameters.get(node_type, [])


def create_empty_node(node_type: str='list', parameter=None):
    ''' A shorthand to create a single AST node of a given type. '''

    #assert(parameter in node_type_parameters.get(node_type, [None]))
    node_type_with_parameter = merge_node_type_parameter(node_type, parameter)
    children = []
    if node_type == 'maze_forever':
        children.append(ASTNode('list'))
    if node_type == 'maze_ifElse':
        children.append(ASTNode('list'))
        children.append(ASTNode('list'))
    return ASTNode(node_type_with_parameter, children)


def dsamepow10(num):
    r = 1
    while num >= 10:
        num = int(num // 10)
        r *= 10
    return r


def join_ints(a, b):
    return a * dsamepow10(b) * 10 + b


def int_to_node(num):
    h = num % 10
    t = num // 10

    type_enum = NodeType(h)
    type_str = node_type_to_str[type_enum]
    
    if t == 0:
        return (ASTNode(type_str, [], node_type_enum=type_enum), 0)

    children = []
    if type_enum != NodeType.LIST:
        while True:
            a = t % 10
            b = t // 10
            if NodeType(a) != NodeType.LIST:
                break
            child, t = int_to_node(t)
            children.append(child)
        return (ASTNode(type_str, children, node_type_enum=type_enum), t)
    else:
        while True:
            a = t % 10
            b = t // 10
            if NodeType(a) == NodeType.LIST_END:
                return (ASTNode(type_str, children, node_type_enum=type_enum), b)
            child, t = int_to_node(t)
            children.append(child)


def ast_to_json(node: ASTNode):
    ''' Converts an ASTNode into a JSON dictionary.
        Works for both HOC4 and HOC18 datasets. '''

    if node.label() == 'list':
        return {
            'type': 'program',
            'children': [ ast_to_json(child) for child in node.children() ],
        }

    node_type, parameter = split_node_type_parameter(node.label())

    node_dict = { 'type': node_type }
    children = []

    if parameter is not None:
        children.append({ 'type': parameter })


    if node_type == 'maze_forever':
        do_dict = { 'type': 'DO' }
        do_list = node.children()[0]
        do_dict['children'] = [ ast_to_json(child) for child in do_list.children() ]
        children.append(do_dict)
    elif node_type == 'maze_ifElse':
        do_dict = { 'type': 'DO' }
        do_list = node.children()[0]
        do_dict['children'] = [ ast_to_json(child) for child in do_list.children() ]
        children.append(do_dict)
        else_dict = { 'type': 'ELSE' }
        else_list = node.children()[1]
        else_dict['children'] = [ ast_to_json(child) for child in else_list.children() ]
        children.append(else_dict)

    if children:
        node_dict['children'] = children
    return node_dict
        

def json_to_ast(root) -> ASTNode:
    ''' Converts a JSON dictionary to an ASTNode.
        Works for both HOC4 and HOC18 datasets. '''

    def get_children(json_node):
        children = json_node.get('children', [])
        # Avoid inconsistencies with lissing statementList
        if len(children) == 1 and children[0]['type'] == 'statementList':
            return get_children(children[0])
        return children

    node_type = root['type']
    children = get_children(root)

    if node_type == 'program':
        return ASTNode('list', [ json_to_ast(child) for child in children ])

    if node_type == 'statementList':
        return ASTNode('list', [ json_to_ast(child) for child in children ])

    if node_type == 'maze_forever':
        do = children[0]
        if do['type'] == 'DO':
            children = get_children(do)
        # else: # This AST is missing 'DO' node for some reason
        return ASTNode(
            node_type,
            [ ASTNode('list', [ json_to_ast(child) for child in children]) ],
        )

    if node_type == 'maze_ifElse':
        assert(len(children) == 3) # Must have condition, do, else nodes for children
        condition = children[0]
        assert(condition['type'] in node_type_parameters[node_type])
        do_node = children[1]
        assert(do_node['type'] == 'DO')
        else_node = children[2]
        assert(else_node['type'] == 'ELSE')
        do_list = [ json_to_ast(child) for child in get_children(do_node) ]
        else_list = [ json_to_ast(child) for child in get_children(else_node) ]
        # node_type is 'maze_ifElse_isPathForward' or 'maze_ifElse_isPathLeft' or 'maze_ifElse_isPathRight'
        return ASTNode(
            merge_node_type_parameter(node_type, condition['type']),
            [ ASTNode('list', do_list), ASTNode('list', else_list) ],
        )

    if node_type == 'maze_moveForward':
        return ASTNode(node_type)

    if node_type == 'maze_turn':
        direction = children[0]
        assert(direction['type'] in node_type_parameters[node_type])
        # node_type is 'maze_turn_turnLeft' or 'maze_turn_turnRight'
        return ASTNode(merge_node_type_parameter(node_type, direction['type']))
    # The following have to be 'maze_turn' + 'turnLeft'/'turnRight' in correct ASTs:
    if node_type == 'maze_turnLeft':
        return ASTNode(merge_node_type_parameter('maze_turn', 'turnLeft'))
    if node_type == 'maze_turnRight':
        return ASTNode(merge_node_type_parameter('maze_turn', 'turnRight'))

    print('Unexpected node type, failing:', node_type)
    assert(False)
    return None


