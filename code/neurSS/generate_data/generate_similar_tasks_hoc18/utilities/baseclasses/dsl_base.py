'''
code based on https://github.com/carpedm20/karel, 
and https://github.com/alts/karel
'''

import random
from functools import wraps
from collections import defaultdict

from ..utils import get_rng


class Dsl(object):
    def __init__(self, rng=None, min_int=0, max_int=19,
            max_func_call=100, debug=False, **kwargs):
        '''
        self.parser is present in the derived class DslHoc or DslKarel.
        Therefore base class Dsl objects should not be created.
        This __init__ method is called in a special way by using the derived class directly
        (not an object of the derived class) to invoke this __init__ of the base class,
        and also passing an object of the derived class as the first self argument 
        (which would have been an object of this base class had super().__init__ been used).
        Check dsl_hoc.py for more info.
        '''

        self.names = {}
        self.debug = kwargs.get('debug', 0)

        self.min_int = min_int
        self.max_int = max_int
        self.max_func_call = max_func_call
        self.int_range = list(range(min_int, max_int+1))

        int_tokens = ['INT{}'.format(num) for num in self.int_range]
        self.tokens_details = list(set(self.parser.tokens) - set(['INT'])) + int_tokens

        self.idx_to_token = { idx: token for idx, token in enumerate(self.parser.tokens) }
        self.token_to_idx = { token:idx for idx, token in self.idx_to_token.items() }

        self.tokens_details.sort()
        self.tokens_details = ['END'] + self.tokens_details

        # vocabulary of tokens to corresponding indices
        self.idx_to_token_details = {
                idx: token for idx, token in enumerate(self.tokens_details) }
        self.token_to_idx_details = {
                token:idx for idx, token in self.idx_to_token_details.items() }

        self.rng = get_rng(rng)
        self.call_counter = [0]

        def callout(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                if self.call_counter[0] > self.max_func_call:
                    raise TimeoutError
                r = f(*args, **kwargs)
                self.call_counter[0] += 1
                return r
            return wrapped

        self.callout = callout





    def lex_to_idx(self, code, details=False):

        tokens = [] 
        self.parser.lexer.input(code)
        while True:
            tok = self.parser.lexer.token()
            if not tok:
                break

            if details:
                if tok.type == 'INT':
                    idx = self.token_to_idx_details["INT{}".format(tok.value)]
                else:
                    idx = self.token_to_idx_details[tok.type]
            else:
                idx = self.token_to_idx[tok.type]
            tokens.append(idx)
        print(self.token_to_idx_details)
        return tokens


    def random_code(self, create_hit_info=False, *args, **kwargs):
        code = " ".join(self.random_tokens(*args, **kwargs))

        # check minimum # of move()
        min_move = getattr(kwargs, 'min_move', 0)
        count_diff = min_move - code.count(self.parser.t_MOVE)

        if count_diff > 0:
            action_candidates = []
            tokens = code.split()

            for idx, token in enumerate(tokens):
                if token in self.parser.action_functions and token != self.parser.t_MOVE:
                    action_candidates.append(idx)

            idxes = self.rng.choice(
                    action_candidates, min(len(action_candidates), count_diff))
            for idx in idxes:
                tokens[idx] = self.parser.t_MOVE
            code = " ".join(tokens)

        if create_hit_info:
            self.hit_info = defaultdict(int)
        else:
            self.hit_info = None

        return code


    def random_tokens(self, start_token="prog", depth=0, stmt_min_depth=2, stmt_max_depth=5, **kwargs):
        if start_token == 'stmt':
            if depth > stmt_max_depth:
                start_token = "action"

        codes = []
        candidates = self.parser.prodnames[start_token]

        prod = candidates[self.rng.randint(len(candidates))]

        for term in prod.prod:
            if term in self.parser.prodnames: # need digging
                codes.extend(self.random_tokens(term, depth + 1, stmt_max_depth))
            else:
                token = getattr(self.parser, 't_{}'.format(term))
                if callable(token):
                    if token == self.parser.t_INT:
                        token = self.parser.random_INT()
                    else:
                        raise Exception(" [!] Undefined token `{}`".format(token))

                codes.append(str(token).replace('\\', ''))

        return codes


    def check_syntax(self, program):
        # create a dummy, large, empty world
        yacc = program.dsl.parser.yacc


