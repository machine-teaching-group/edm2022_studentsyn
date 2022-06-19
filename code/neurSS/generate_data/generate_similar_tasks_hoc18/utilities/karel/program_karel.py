'''
code based on https://github.com/carpedm20/karel, 
and https://github.com/alts/karel
'''

from ..baseclasses.program_base import Program


class ProgramKarel(Program):
    def __init__(self, code=None, dsl=None):
        super().__init__(code, dsl)