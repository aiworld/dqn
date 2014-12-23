# [x] download s3 files
# [x] download firebase votes for each episode
# [x] integrate reward into experiences
# [x] populate experience pairs with human labeled experiences
# [x] run as normal, except don't overwrite experiences
# [x] load experience pairs in parallel
# [x] don't cache experience pairs
# [x] see how large experience pairs are normally (non-integration)
# [x] lower the momentum
# [ ] create experience deque that separate process pushes to, and solver pulls from.
# [ ] solver should ideally have a thread that loads the next experience minibatch from the deque
# [ ] increase the minibatch pool by
# [ ] turn momentum and learning rate back up
import json
import os
import threading
import psutil
import random
import snappy

from boto.s3.connection import S3Connection
import time
from constants import DQN_ROOT, VOTE_URL, FIREBASE_URL, INTEGRATE_DIR
from secrets import ADMIN_EMAIL, ADMIN_PASSWORD, FIREBASE_KEY
import secrets

from firebase import firebase as fb
import gzip

FETCH_EPISODES = False
EXPERIENCE_CACHE = {}
USE_CACHE = False


def get_episodes():
    conn = S3Connection(secrets.DQN_AWS_ID, secrets.DQN_AWS_SECRET)
    bucket = conn.get_bucket('aiworld')
    episodes = list(bucket.list())
    ret = []
    for episode in episodes:
        if episode.key.find('1414651242') >= 0:  # TODO: Support all batches.
            ret.append(episode)
    return ret


def store_integrated_experiences():
    episodes = get_episodes()
    auth = fb.FirebaseAuthentication(FIREBASE_KEY, ADMIN_EMAIL, ADMIN_PASSWORD)
    fire = fb.FirebaseApplication(FIREBASE_URL, auth)
    i = 0
    mem_pct = psutil.phymem_usage().percent
    while i < len(episodes) and mem_pct < 100:
        print str(i) + ' ' + str(len(episodes))
        episode = episodes[i]
        episode_directory, episode_number = episode.key.split('/')
        pre_dir  = DQN_ROOT + '/data/s3/episodes/'        + episode_directory
        post_dir = INTEGRATE_DIR + episode_directory
        if not os.path.exists(pre_dir):
            os.makedirs(pre_dir)
        pre_filename  = pre_dir        + episode_number
        post_filename = post_dir + '_' + episode_number + '.snappy'
        if not os.path.exists(pre_filename):
            episode.get_contents_to_filename(pre_filename)
        if os.path.exists(post_filename):
            print post_filename + ' already loaded'
        else:
            save_snappy_file_pll(episode_directory, episode_number, fire,
                             post_filename, pre_filename)
        i += 1


def save_snappy_file_pll(episode_directory, episode_number, fire,
                             post_filename, pre_filename):
    threading.Thread(target=save_snappy_file, args=(episode_directory,
                            episode_number, fire, post_filename, pre_filename))


def save_snappy_file(episode_directory, episode_number, fire, post_filename,
                     pre_filename):
    with gzip.GzipFile(pre_filename, 'r', 6) as pre_data:
        javascript = pre_data.read()
        json_str = javascript[javascript.index('=') + 1:].strip()
        episode_data = json.loads(json_str)
    votes = fire.get(VOTE_URL + '/' + episode_directory, episode_number)
    # Integrate
    # (image_action, action, game_over, reward, votes)
    if votes:
        experiences = combine(votes, episode_data)
        with open(post_filename, 'w', 6) as post_data:
            post_data.write(snappy.compress(json.dumps(experiences)))


def add_votes_property(frames):
    for frame in frames:
        for sub_frame in frame:
            sub_frame['votes'] = 0


def combine(votes, episode_data):
    frames = episode_data['frames']
    add_votes_property(frames)
    for vote in votes.values():
        sub_frame = frames[vote['frame']][vote['subFrame']]
        if vote['good']:
            sub_frame['reward'] += 1
        else:
            sub_frame['reward'] -= 1
    return frames


# def get_random_experience_pairs():
#     filename = random.choice(os.listdir(INTEGRATE_DIR))
#     ret = None
#     if not filename.startswith('.'):
#         print 'integrate file:', filename
#         ret = get_experience_pairs(filename)
#     if not ret:
#         return get_random_experience_pairs()
#     else:
#         return ret


def get_all_experience_pairs():
    filenames = [n for n in os.listdir(INTEGRATE_DIR) if not n.startswith('.')]
    load_experience_files_in_parallel(filenames)


def load_experience_files_in_parallel(filenames):
    ret = []
    threads = []

    def worker(filename, i):
        """thread worker function"""
        print 'loading...' + filename + ' ' + str(i)
        ret.append(get_experience_pairs(filename))
        mem_pct = psutil.phymem_usage().percent
        print 'done with ' + filename + ' ' + str(i) + ' mem ' + str(mem_pct)
        return

    for i, fn in enumerate(filenames):
        t = threading.Thread(target=worker, args=(fn, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return ret


def get_experience_pairs(filename):
    if USE_CACHE and filename in EXPERIENCE_CACHE:
        print 'EXPERIENCE_CACHE hit!'
        json_str = read_cache(filename)
    else:
        if USE_CACHE:
            print 'EXPERIENCE_CACHE miss!'
        with open(INTEGRATE_DIR + filename, 'r') as file_ref:
            json_str = snappy.decompress(file_ref.read())
        if USE_CACHE and psutil.phymem_usage().percent < 80:
            write_cache(filename, json_str)
        else:
            if USE_CACHE:
                print 'EXPERIENCE_CACHE FULL!'
    experiences = json.loads(json_str)
    # compress_screen_hex(experiences)
    return pair_experiences(experiences, filename)


def write_cache(filename, json_str):
    EXPERIENCE_CACHE[filename] = snappy.compress(json_str)
    print 'EXPERIENCE_CACEE length', len(EXPERIENCE_CACHE)


def read_cache(filename):
    return snappy.decompress(EXPERIENCE_CACHE[filename])


def compress_screen_hex(experiences):
    for e in experiences:
        for f in e:
            f['screen_hex'] = snappy.compress(f['screen_hex'])

# def pair_experiences(experiences, filename):
#     global total_pairs
#     ret = []
#     exp_len = len(experiences)
#     inc_total_pairs_and_games(exp_len)
#     print 'pairs len: ' + str(exp_len)
#     for i in xrange(0, exp_len, 2):
#         pair = experiences[i: i + 2]
#         if len(pair) != 2:
#             print 'pair not right:', filename, i
#         else:
#             ret.append(pair)
#     return ret


def test_snappy():
    # store_integrated_experiences()
    test = []
    for i in xrange(35):
        print i
        test.append(get_random_experience_pairs())
    print time.time()
    for t in test:
        for p in t:
            for e in p:
                for f in e:
                    snappy.decompress(f['screen_hex'])


if __name__ == '__main__':
    # store_integrated_experiences()
    get_all_experience_pairs()
