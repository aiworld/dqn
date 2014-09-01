from collections import OrderedDict
import random


class Action(object):
    def __init__(self, value, up, down, right, left, fire, name):
        self.value   = value
        self.up      = up
        self.down    = down
        self.right   = right
        self.left    = left
        self.fire    = fire
        self.name    = name
        self.index   = None

    def __repr__(self):
        return str(self.value)

"""                              'value' , 'up'   , 'down' , 'right' , 'left'  , 'fire', 'name'])                  """
NOOP                     = Action ( 0    ,  False ,  False ,  False  ,  False ,  False , 'NOOP'                     )
FIRE                     = Action ( 1    ,  False ,  False ,  False  ,  False ,  True  , 'FIRE'                     )
MOVE_UP                  = Action ( 2    ,  True  ,  False ,  False  ,  False ,  False , 'MOVE_UP'                  )
MOVE_RIGHT               = Action ( 3    ,  False ,  False ,  True   ,  False ,  False , 'MOVE_RIGHT'               )
MOVE_LEFT                = Action ( 4    ,  False ,  False ,  False  ,  True  ,  False , 'MOVE_LEFT'                )
MOVE_DOWN                = Action ( 5    ,  False ,  True  ,  False  ,  False ,  False , 'MOVE_DOWN'                )
MOVE_UP_AND_RIGHT        = Action ( 6    ,  True  ,  False ,  True   ,  False ,  False , 'MOVE_UP_AND_RIGHT'        )
MOVE_UP_AND_LEFT         = Action ( 7    ,  True  ,  False ,  False  ,  True  ,  False , 'MOVE_UP_AND_LEFT'         )
MOVE_DOWN_AND_RIGHT      = Action ( 8    ,  False ,  True  ,  True   ,  False ,  False , 'MOVE_DOWN_AND_RIGHT'      )
MOVE_DOWN_AND_LEFT       = Action ( 9    ,  False ,  True  ,  False  ,  True  ,  False , 'MOVE_DOWN_AND_LEFT'       )
MOVE_UP_AND_FIRE         = Action ( 10   ,  True  ,  False ,  False  ,  False ,  True  , 'MOVE_UP_AND_FIRE'         )
MOVE_RIGHT_AND_FIRE      = Action ( 11   ,  False ,  False ,  True   ,  False ,  True  , 'MOVE_RIGHT_AND_FIRE'      )
MOVE_LEFT_AND_FIRE       = Action ( 12   ,  False ,  False ,  False  ,  True  ,  True  , 'MOVE_LEFT_AND_FIRE'       )
MOVE_DOWN_AND_FIRE       = Action ( 13   ,  False ,  True  ,  False  ,  False ,  True  , 'MOVE_DOWN_AND_FIRE'       )
MOVE_UP_RIGHT_AND_FIRE   = Action ( 14   ,  True  ,  False ,  True   ,  False ,  True  , 'MOVE_UP_RIGHT_AND_FIRE'   )
MOVE_UP_LEFT_AND_FIRE    = Action ( 15   ,  True  ,  False ,  False  ,  True  ,  True  , 'MOVE_UP_LEFT_AND_FIRE'    )
MOVE_DOWN_RIGHT_AND_FIRE = Action ( 16   ,  False ,  True  ,  True   ,  False ,  True  , 'MOVE_DOWN_RIGHT_AND_FIRE' )
MOVE_DOWN_LEFT_AND_FIRE  = Action ( 17   ,  False ,  True  ,  False  ,  True  ,  True  , 'MOVE_DOWN_LEFT_AND_FIRE'  )
RESET                    = Action ( 40   ,  False ,  False ,  False  ,  False ,  False , 'RESET'                    )

# Just actions relevant to Space Invaders for now.
ALL = OrderedDict()
for i, a in enumerate([FIRE, MOVE_RIGHT_AND_FIRE, MOVE_LEFT_AND_FIRE]):
    a.index = i
    ALL[a.name] = a

if __name__ == '__main__':
    print ALL


def get_random_action():
    return ALL.values()[random.randint(0, len(ALL.values()) - 1)]
