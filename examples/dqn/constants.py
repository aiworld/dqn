import os
LAYER_NAMES              = ['conv1', 'conv2', 'fc1', 'fc2']
CAFFE_ROOT               = '/s/caffe/'
DQN_ROOT                 = os.path.dirname(os.path.realpath(__file__))
EPISODE_DIR_NAME         = 'episodes'
INTEGRATE_HUMAN_FEEDBACK = 'INTEGRATE_HUMAN_FEEDBACK' in os.environ
FIREBASE_URL             = 'https://vivid-fire-9851.firebaseio.com'
VOTE_URL                 = FIREBASE_URL + '/votes'
BATCH_LIST_URL           = FIREBASE_URL + '/batches'
EPISODE_LIST_URL         = FIREBASE_URL + '/episodes'
