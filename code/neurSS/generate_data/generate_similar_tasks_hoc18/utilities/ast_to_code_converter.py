'''
command:
python -m tasksynthesis.utilities.utility.ast_to_code_converter --data_in_path=../data/ignore/data/in_tasks/tasks/ --data_out_path=../data/ignore/data/in_tasks/tasks_code/

reference: https://github.com/bunelr/GandRL_for_NPS/blob/master/karel/ast_converter.py
'''

import argparse
import json
import os


class AstToCodeConverter(object):
    def __init__(self):
        actions = [
            'move',
            'turn_left',
            'turn_right',
            'pick_marker',
            'put_marker'
        ]
        self.action_hash = {}
        for x in actions:
            self.action_hash[x] = None
        
        conditionals = [
            'bool_goal',
            'bool_path_ahead',
            'bool_no_path_ahead',
            'bool_path_left',
            'bool_no_path_left',
            'bool_path_right',
            'bool_no_path_right',
            'bool_marker',
            'bool_no_marker'
        ]
        self.conditional_hash = {}
        for x in conditionals:
            self.conditional_hash[x] = None


    def to_tokens(self, ast):
        '''
        Karel code is allowed to have multiple methods defined. In our case, we only have
        a single run method. If other methods were present, this function will iteratively
        tokenize each method by calling _make_method.
        '''
        tokens = []

        # add run method
        assert ast["type"] == "run", "ast doesn't contain run method"
        method_name = ast["type"]
        method_json = ast["children"]
        self._make_method(method_name, method_json, tokens)

        return tokens


    def _make_method(self, name, json, tokens):
        tokens.append('DEF')
        tokens.append(name)
        tokens.append('m(')
        self._expand_code_block(json, tokens)
        tokens.append('m)')


    def _expand_code_block(self, code_block, tokens):
        for block in code_block:
            #print(block)
            block_type = block['type']

            # basic actions
            if( self._is_action(block_type) ):
                tokens.append(block_type)
            # for loops
            elif( ("repeat_until" not in block_type) and ("repeat" in block_type) ):
                num_times = block_type.split('(')[1][:-1]
                body = block['children']

                tokens.append('REPEAT')
                tokens.append('R={0}'.format(num_times))
                tokens.append('r(')
                self._expand_code_block(body, tokens)
                tokens.append('r)')
            # while loops
            # karel dsl has while and hoc dsl has repeat_until_goal
            elif( ("while" in block_type) or ("repeat_until_goal" in block_type) ):
                body = block['children']
                tokens.append('WHILE')
                self._expand_condition_block(block_type, tokens)
                tokens.append('w(')
                self._expand_code_block(body, tokens)
                tokens.append('w)')
            # if statements
            elif( ("ifelse" not in block_type) and ("if" in block_type) ):
                # children contained inside "do" node within "if" node
                if_block = block['children'][0]
                assert if_block["type"] == "do", "children of if not contained inside do ast node"
                body = if_block["children"]
                
                tokens.append('IF')
                self._expand_condition_block(block_type, tokens)
                tokens.append('i(')
                self._expand_code_block(body, tokens)
                tokens.append('i)')
            # if/else statements
            elif( "ifelse" in block_type ):
                if_block = block['children'][0]
                else_block = block['children'][1]
                if_body = if_block['children']
                else_body = else_block['children']

                tokens.append('IFELSE')
                self._expand_condition_block(block_type, tokens)
                tokens.append('i(')
                self._expand_code_block(if_body, tokens)
                tokens.append('i)')

                tokens.append('ELSE')
                tokens.append('e(')
                self._expand_code_block(else_body, tokens)
                tokens.append('e)')
            # parse error
            else:
                raise Exception('unknown type: \''+block_type+'\'')


    def _expand_condition_block(self, block_type, tokens):
        tokens.append('c(')
        conditional = block_type.split("(")[1][:-1]
        if( self._is_conditional(conditional) ):
            # unify goal in hoc as no_marker in karel
            if( conditional == "bool_goal" ):
                conditional = "bool_no_marker"
            tokens.append(conditional)
        tokens.append('c)')


    def _is_conditional(self, condition):
        return condition in self.conditional_hash


    def _is_action(self, block_type):
        
        return block_type in self.action_hash


def convert_files(data_in_path, data_out_path):
    converter = AstToCodeConverter()

    for filename in os.listdir(args.data_in_path):
        if( "code" in filename ):
            print("Converting {} from ast to code".format(filename))
            ast_in_path = os.path.join(args.data_in_path, filename)
            with open(ast_in_path) as f:
                ast = json.load(f)

            #print("ast:\n", ast)
            tokens = converter.to_tokens(ast)
            code = " ".join(tokens)
            #print("tokens:\n", tokens)
            filename_txt = filename.split(".")[0]+".txt"
            code_out_path = os.path.join(args.data_out_path, filename_txt)
            with open(code_out_path, "w") as f:
                f.write(code)


def convert_ast_to_code(ast):
    converter = AstToCodeConverter()
    tokens = converter.to_tokens(ast)
    code = " ".join(tokens)

    return code


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_in_path', type=str, default=None, help='Path to input data directory')
    arg_parser.add_argument('--data_out_path', type=str, default=None, help='Path to output directory to store converted code files')
    args = arg_parser.parse_args()

    if not os.path.exists(args.data_out_path):
        os.makedirs(args.data_out_path)

    convert_files(args.data_in_path, args.data_out_path)


