# [x] download s3 files
# [x] download firebase votes for each episode
# [x] integrate reward into experiences
# [x] populate experience pairs with human labeled experiences
# [x] run as normal, except don't overwrite experiences
# [ ] load experience pairs in parallel
# [x] don't cache experience pairs
# [x] see how large experience pairs are normally (non-integration)
# [ ] lower the momentum
import json
import os
import psutil
import random
import snappy

from boto.s3.connection import S3Connection
from constants import DQN_ROOT, VOTE_URL, FIREBASE_URL
from secrets import ADMIN_EMAIL, ADMIN_PASSWORD, FIREBASE_KEY
import secrets

from firebase import firebase as fb
import gzip

FETCH_EPISODES = False
INTEGRATE_DIR = DQN_ROOT + '/data/integrated/episodes/'
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
        print mem_pct
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
        with gzip.GzipFile(pre_filename, 'r', 6) as pre_data:
            javascript = pre_data.read()
            json_str = javascript[javascript.index('=') + 1 :].strip()
            episode_data = json.loads(json_str)
        votes = fire.get(VOTE_URL + '/' + episode_directory, episode_number)
        # Integrate
        # (image_action, action, game_over, reward, votes)
        if votes:
            experiences = combine(votes, episode_data)
            with open(post_filename, 'w', 6) as post_data:
                post_data.write(snappy.compress(json.dumps(experiences)))
        i += 1


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


def get_random_experience_pairs():
    filename = random.choice(os.listdir(INTEGRATE_DIR))
    ret = None
    if not filename.startswith('.'):
        print 'integrate file:', filename
        ret = get_experience_pairs(filename)
    if not ret:
        return get_random_experience_pairs()
    else:
        return ret


def get_experience_pairs(filename):
    if USE_CACHE and filename in EXPERIENCE_CACHE:
        print 'EXPERIENCE_CACHE hit!'
        json_str = read_cache(filename)
    else:
        print 'EXPERIENCE_CACHE miss!'
        with open(INTEGRATE_DIR + filename, 'r') as file_ref:
            json_str = snappy.decompress(file_ref.read())
        if USE_CACHE and psutil.phymem_usage().percent < 80:
            write_cache(filename, json_str)
        else:
            print 'EXPERIENCE_CACHE FULL!'
    return pair_experiences(json_str, filename)


def write_cache(filename, json_str):
    EXPERIENCE_CACHE[filename] = snappy.compress(json_str)
    print 'EXPERIENCE_CAHCE length', len(EXPERIENCE_CACHE)


def read_cache(filename):
    return snappy.decompress(EXPERIENCE_CACHE[filename])


def pair_experiences(json_str, filename):
    ret = []
    experiences = json.loads(json_str)
    exp_len = len(experiences)
    print 'pairs len:', exp_len
    for i in xrange(0, exp_len, 2):
        pair = experiences[i: i + 2]
        if len(pair) != 2:
            print 'pair not right:', filename, i
        else:
            ret.append(pair)
    return ret

if __name__ == '__main__':
    store_integrated_experiences()
    # get_random_experience_pairs()
