import os
LAYER_NAMES              = ['conv1', 'conv2', 'fc1', 'fc2']
CAFFE_ROOT               = '/s/caffe/'
DQN_ROOT                 = os.path.dirname(os.path.realpath(__file__))
EPISODE_DIR_NAME         = 'episodes'
INTEGRATE_HUMAN_FEEDBACK = 'INTEGRATE_HUMAN_FEEDBACK' in os.environ
PLOT_LAYERS              = 'PLOT_LAYERS'              in os.environ
FIREBASE_URL             = 'https://vivid-fire-9851.firebaseio.com'
VOTE_URL                 = FIREBASE_URL + '/votes'
BATCH_LIST_URL           = FIREBASE_URL + '/batches'
EPISODE_LIST_URL         = FIREBASE_URL + '/episodes'
INTEGRATE_DIR            = DQN_ROOT + '/data/integrated/episodes/'
EXP_IMAGE_ACTION_INDEX   = 0
EXP_ACTION_INDEX         = 1
EXP_GAME_OVER_INDEX      = 2
EXP_REWARD_INDEX         = 3
EXP_SCREEN_HEX_INDEX     = 4
