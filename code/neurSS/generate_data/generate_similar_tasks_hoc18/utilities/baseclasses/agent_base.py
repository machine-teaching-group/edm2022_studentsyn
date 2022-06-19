'''
Code based on https://github.com/carpedm20/karel, 
and https://github.com/alts/karel
'''

class _Agent(object):
    '''
    '''
    def __init__(self, position, facing):
        self.position = position
        self.facing = facing


    def move(self):
        self.position = (
            self.position[0] + self.facing[0],
            self.position[1] + self.facing[1]
        )


    def turn_left(self):
        self.facing = (
            self.facing[1],
            -self.facing[0]
        )

    def turn_right(self):
        self.facing = (
            -self.facing[1],
            self.facing[0]
        )