from .parser.parser_tree import ParseToTree
import itertools

'''
Convert between different types of student code representations
'''

def ast_code_to_token_list(ast):
    '''
        benchmark ast -> token list 
        data ast -> token list 
    '''
    res = ast_to_tokens_recursive(ast)
    return list(map(lambda x : tokens_ast_to_list.get(x,x), res))

def benchmark_code_to_student_code(code_ast):
    '''
        benchmark ast -> data ast 
    '''
    children = [convert_recursive(child) for child in code_ast["children"]]
    root_node = {"type" : "program", 
                 "children" : children}
    return root_node 

def token_list_to_benchhmark_ast(code_tokens):
    '''
        token list -> benchmark ast
    '''
    code_str = ' '.join(code_tokens)
    ast = ParseToTree().parse(code_str)
    return ast 


def flatten_list2d(list2d):
    '''
        flatten a list a single time 
    '''
    return list(itertools.chain(*list2d))

def ast_to_tokens_recursive(ast):
    if ast["type"] in ["program","run"] : 
        res = flatten_list2d([ast_to_tokens_recursive(a) for a in ast.get('children',[]) ])
        res = ["DEF","run","m("] + res + ["m)"]

    elif ast["type"] in ["maze_forever","while"] : 

        res = flatten_list2d([ast_to_tokens_recursive(a) for a in ast.get('children',[]) ])
        res = ["WHILE","c(","bool_no_marker","c)","w("] + res + ["w)"]

    elif ast["type"] == "maze_ifElse" :

        ast_else = None 

        for a in ast["children"]:
            if a["type"] in ["isPathForward","isPathRight","isPathLeft"]:
                ast_cond = a 
            elif a["type"] == "DO" : 
                ast_if = a 
            elif a["type"] == "ELSE" :
                ast_else = a  

        res = ["IFELSE","c("] + [ast_cond["type"]] + ["c)"] + ["i("]
        res = res + ast_to_tokens_recursive(ast_if) + ["i)"]

        if ast_else != None : 
            res = res + ["ELSE"] + ["e("] + ast_to_tokens_recursive(ast_else) + ["e)"]
   
   
    # handle second type of ifelse notation
    elif ast["type"].split()[0] in ['ifelse','if']:
        cond = ast["type"].split()[-1]

        for a in ast["children"]:
            if a["type"] == "do" : 
                ast_if = a 
            elif a["type"] == "else" :
                ast_else = a  

        res = ["IFELSE","c("] + [cond] + ["c)"] + ["i("]
        res = res + ast_to_tokens_recursive(ast_if) + ["i)"]
        if ast_else != None : 
            res = res + ["ELSE"] + ["e("] + ast_to_tokens_recursive(ast_else) + ["e)"]

    elif ast["type"] in ["turnLeft","maze_moveForward","turnRight",
                "turn_left", "move", "turn_right", "<pad>"]:
        res = [ast["type"]]

    else : 
        # ignore node 
        res = flatten_list2d([ast_to_tokens_recursive(a) for a in ast.get('children',[]) ])

    return res 
    


def convert_recursive(code_ast): 
    node_type, node_token = seperate_token_from_type(code_ast["type"])
    translate_fn = lambda token : tokens_benchmark_to_student.get(token, token)
    node_type = translate_fn(node_type)
    node_token = translate_fn(node_token)

    children = []
    if node_token is not None : 
        child_node = {"type" : node_token}
        children.append(child_node)
    
    if node_type == 'maze_forever':
        do_dict = { 'type': 'DO' }
        do_dict['children'] = [ convert_recursive(child) for child in code_ast["children"] ]
        children.append(do_dict)

    elif node_type == 'maze_ifElse':
        children = children + [convert_recursive(child) for child in code_ast["children"]]

    elif node_type in ["DO","ELSE"] : 
        children = [convert_recursive(child) for child in code_ast["children"]]

    node = {"type" : node_type, "children" : children}
    if children:
        node['children'] = children
 
    return node

tokens_benchmark_to_student = {
    "turn_right" : "turnRight", 
    "turn_left" : "turnLeft",
    "move" : "maze_moveForward",
    "bool_path_ahead" : "isPathForward" ,
    "bool_path_left" : "isPathLeft",
    "bool_path_right" : "isPathRight",
    "while" : "maze_forever",
    "ifelse" : "maze_ifElse",
    "do" : "DO",
    "else" : "ELSE"
}

tokens_ast_to_list = {
        "turnRight" : "turn_right",
        "turnLeft" : "turn_left",
        "maze_moveForward" : "move",
        "is_path_forward" : "bool_path_ahead",
        "is_path_left" : "bool_path_left",
        "is_path_right" : "bool_path_right",
        "isPathForward" : "bool_path_ahead",
        "isPathRight" : "bool_path_right", 
        "isPathLeft" : "bool_path_left"
    }



def seperate_token_from_type(t):
    if t in ["turn_right","turn_left"]:
        return "maze_turn",t
    elif 'ifelse' in t : 
        return t.split()[0], t.split()[1]
    else :
        return t, None 

