'''
code based on https://github.com/carpedm20/karel, 
and https://github.com/alts/karel
'''
class _Task(object):
    def __init__(self, world=None, agent=None):
        self.world = world
        self.agent = agent
