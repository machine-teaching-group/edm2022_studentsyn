'''
code based on https://github.com/carpedm20/karel, 
and https://github.com/alts/karel
'''

from .parser_karel_unified import _ParserKarelUnified
from .program_karel import ProgramKarel
from ..baseclasses.dsl_base import Dsl


class DslKarel(Dsl):
    def __init__(self, rng=None, min_int=0, max_int=19,
            max_func_call=100, debug=False, **kwargs):

        ########################
        # Build lexer and parser
        ########################
        self.parser = _ParserKarelUnified(rng, min_int, max_int, debug)
        Dsl.__init__( 
                self, rng, min_int, max_int,
                max_func_call, debug, **kwargs
        )


    def random_code(self, create_hit_info=False, *args, **kwargs):
        code = super().random_code(create_hit_info, *args, **kwargs)
        program = ProgramKarel(code)

        return program