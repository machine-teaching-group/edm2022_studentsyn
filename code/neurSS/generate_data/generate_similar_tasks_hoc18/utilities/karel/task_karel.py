'''
code based on https://github.com/carpedm20/karel, 
and https://github.com/alts/karel
'''

from ..baseclasses.task_base import _Task
import numpy as np


class _TaskKarel(_Task):
    def __init__(self, world=None, agent=None, agent_direction=None, markers=[], max_marker_in_cell=1):
        super().__init__(world, agent)
        self.markers = markers
        state = np.zeros_like(self.world, dtype=np.int8)
        self.zero_state = np.tile(
                np.expand_dims(state, -1),
                [1, 1, agent_direction + 1 + (max_marker_in_cell + 1)])



