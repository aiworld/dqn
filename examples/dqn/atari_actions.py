from collections import namedtuple
import random
Action = namedtuple('Action', [  'value' , 'up'   , 'down' , 'right' , 'left'  , 'fire'])
NOOP                     = Action ( 0    ,  False , False  ,  False  , False   , False )
FIRE                     = Action ( 1    ,  False , False  ,  False  , False   , True  )
MOVE_UP                  = Action ( 2    ,  True  , False  ,  False  , False   , False )
MOVE_RIGHT               = Action ( 3    ,  False , False  ,  True   , False   , False )
MOVE_LEFT                = Action ( 4    ,  False , False  ,  False  , True    , False )
MOVE_DOWN                = Action ( 5    ,  False , True   ,  False  , False   , False )
MOVE_UP_AND_RIGHT        = Action ( 6    ,  True  , False  ,  True   , False   , False )
MOVE_UP_AND_LEFT         = Action ( 7    ,  True  , False  ,  False  , True    , False )
MOVE_DOWN_AND_RIGHT      = Action ( 8    ,  False , True   ,  True   , False   , False )
MOVE_DOWN_AND_LEFT       = Action ( 9    ,  False , True   ,  False  , True    , False )
MOVE_UP_AND_FIRE         = Action ( 10   ,  True  , False  ,  False  , False   , True  )
MOVE_RIGHT_AND_FIRE      = Action ( 11   ,  False , False  ,  True   , False   , True  )
MOVE_LEFT_AND_FIRE       = Action ( 12   ,  False , False  ,  False  , True    , True  )
MOVE_DOWN_AND_FIRE       = Action ( 13   ,  False , True   ,  False  , False   , True  )
MOVE_UP_RIGHT_AND_FIRE   = Action ( 14   ,  True  , False  ,  True   , False   , True  )
MOVE_UP_LEFT_AND_FIRE    = Action ( 15   ,  True  , False  ,  False  , True    , True  )
MOVE_DOWN_RIGHT_AND_FIRE = Action ( 16   ,  False , True   ,  True   , False   , True  )
MOVE_DOWN_LEFT_AND_FIRE  = Action ( 17   ,  False , True   ,  False  , True    , True  )
# RESET                    = Action ( 40   ,  False , False  ,  False  , False   , False )
_props = dict(locals())
ALL = dict([(k, v) for k, v in _props.iteritems() if type(v) is Action])

if __name__ == '__main__':
    print ALL


def get_random_action():
    return ALL.values()[random.randint(0, len(ALL.values()) - 1)]
