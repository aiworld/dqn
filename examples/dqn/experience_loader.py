import json
import multiprocessing
import traceback
import os
import random
import snappy
from constants import *
import atari_actions

NUM_PRODUCERS = 2


def producer(q):
    try:
        while True:
            filename = random.choice(os.listdir(INTEGRATE_DIR))
            if not filename.startswith('.'):
                print 'integrate file: ' + filename
                pairs = load_experience_pairs(filename)
                for pair in pairs:
                    q.put(pair)  # blocks when q is full
                print 'experience queue size is ' + str(q.qsize())
    except:
        # Background process exceptions don't bubble up.
        print "FATAL: load experiences worker exited while multiprocessing"
        traceback.print_exc()


def get_queue():
    mgr = multiprocessing.Manager()
    q = mgr.Queue(maxsize=1000)
    for _ in xrange(NUM_PRODUCERS):
        multiprocessing.Process(
            target=producer, args=(q,)).start()
    return q


def load_experience_pairs(filename):
    with open(INTEGRATE_DIR + filename, 'r') as file_ref:
        json_str = snappy.decompress(file_ref.read())
    print 'finished decompressing ' + filename
    experiences = json.loads(json_str)
    # compress_screen_hex(experiences)
    return pair_experiences(experiences, filename)


def pair_experiences(experiences, filename):
    ret = []
    exp_len = len(experiences)
    print 'pairs len: ' + str(exp_len)
    for i in xrange(0, exp_len, 2):
        pair = experiences[i: i + 2]
        if len(pair) != 2:
            print 'pair not right:', filename, i
        else:
            ret.append([deserialize(pair[0]),
                        deserialize(pair[1])])
    return ret


def deserialize(frames):
    ret = []
    for exp in frames:
        r = [None] * 5
        r[EXP_IMAGE_ACTION_INDEX] =                   exp['image_action']
        r[EXP_ACTION_INDEX]       = atari_actions.ALL[exp['action']]
        r[EXP_GAME_OVER_INDEX]    =                   exp['game_over']
        r[EXP_REWARD_INDEX]       =                   exp['reward']
        r[EXP_SCREEN_HEX_INDEX]   =                   exp['screen_hex']
        ret.append(r)
    return ret

if __name__ == '__main__':
    test_mgr = multiprocessing.Manager()
    test_q = test_mgr.Queue(maxsize=1000)
    producer(test_q)
