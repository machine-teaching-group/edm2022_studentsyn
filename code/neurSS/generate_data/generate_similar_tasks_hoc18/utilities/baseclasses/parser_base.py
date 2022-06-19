'''
code based on https://github.com/carpedm20/karel, 
and https://github.com/alts/karel
'''

import ply.lex as lex

from .. import yacc
from ..utils import get_rng


class _Parser(object):
    tokens = ()
    precedence = ()

    def __init__(self, rng=None, min_int=0, max_int=19, debug=False, env=None, **kwargs):
        # Build the lexer and parser
        modname = self.__class__.__name__
        self.lexer = lex.lex(module=self, debug=debug)
        self.yacc, self.grammar = yacc.yacc(
                module=self,
                debug=debug,
                tabmodule="_parsetab",
                with_grammar=True)

        self.prodnames = self.grammar.Prodnames

        self.min_int = min_int
        self.max_int = max_int
        self.rng = get_rng(rng)

        # reference to env object - required when a program needs to be executed on a task
        # can be None the rest of the time
        self.env = None

        self.flush_hit_info()


    def flush_hit_info(self):
        self.hit_info = {}
        self.funct_table = {} # save parsed function
        self.counter = 0
        self.symbolic_decisions = []


def dummy():
    pass

