import json
import multiprocessing
from threading import Thread
from Queue import Queue
import traceback
import random
import time
import snappy
from constants import *
import atari_actions
from constants import MINIBATCH_SIZE

MAX_QUEUE_SIZE = 2
NUM_PRODUCERS  = 2


def producer(q):
    try:
        while True:
            filename = random.choice(os.listdir(INTEGRATE_DIR))
            if not filename.startswith('.'):
                print 'integrate file: ' + filename
                pairs = load_experience_pairs(filename)
                # if q full, add batch to standby
                # else, put batch into q
                diff = 0
                for batch in chunks(pairs, MINIBATCH_SIZE):
                    # time.sleep(diff)
                    time1 = time.time()
                    q.put(batch)  # blocks when q is full
                    time2 = time.time()
                    diff = time2 - time1
                    print '%s function took %0.3f ms' %\
                          ('queue-put', diff * 1000.0)
                    print 'experience queue size is ' + str(q.qsize())
    except:
        # Background process exceptions don't bubble up.
        print "FATAL: load experiences worker exited while multiprocessing"
        traceback.print_exc()


def get_queue():
    q = Queue(maxsize=MAX_QUEUE_SIZE)
    for _ in xrange(NUM_PRODUCERS):
        worker = Thread(target=producer, args=(q,))
        worker.setDaemon(True)
        worker.start()
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


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

if __name__ == '__main__':
    test_mgr = multiprocessing.Manager()
    test_q = test_mgr.Queue(maxsize=MAX_QUEUE_SIZE)
    producer(test_q)
