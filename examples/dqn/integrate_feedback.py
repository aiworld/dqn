"""
One off script to integrate crowdsourced rewards from firebase with
episode recordings on s3.
"""
# [x] download s3 files
# [x] download firebase votes for each episode
# [x] integrate reward into experiences
# [x] populate experience pairs with human labeled experiences
# [x] run as normal, except don't overwrite experiences
# [x] load experience pairs in parallel
# [x] don't cache experience pairs
# [x] see how large experience pairs are normally (non-integration)
# [x] lower the momentum
# [x] create experience deque that separate process pushes to, and solver pulls from.
# [x] solver should ideally have a thread that loads the next experience minibatch from the deque
# [ ] increase the minibatch pool by ?
# [x] turn momentum and learning rate back up
import json
import os
import threading
import psutil
import random
import snappy
import numpy
from boto.s3.connection import S3Connection
import time

from constants import DQN_ROOT, VOTE_URL, FIREBASE_URL, INTEGRATE_DIR
from secrets import ADMIN_EMAIL, ADMIN_PASSWORD, FIREBASE_KEY, DQN_AWS_ID, DQN_AWS_SECRET

from firebase import firebase as fb
import gzip

FETCH_EPISODES = False
EXPERIENCE_CACHE = {}
USE_CACHE = False


def store_integrated_experiences(parallel=True):
    episodes = get_episodes()
    auth = fb.FirebaseAuthentication(FIREBASE_KEY, ADMIN_EMAIL, ADMIN_PASSWORD)
    fire = fb.FirebaseApplication(FIREBASE_URL, auth)
    i = 0
    mem_pct = psutil.phymem_usage().percent
    while i < len(episodes) and mem_pct < 100:
        episode = episodes[i]
        episode_directory, episode_number = episode.key.split('/')
        print str(i) + ' of ' + str(len(episodes)) + ' #' + episode_number
        pre_dir  = DQN_ROOT + '/data/s3/episodes/' + episode_directory
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
            if parallel:
                save_snappy_file_pll(episode_directory, episode_number, fire,
                             post_filename, pre_filename)
            else:
                save_snappy_file(episode_directory, episode_number, fire,
                             post_filename, pre_filename)
        i += 1


def get_episodes():
    conn = S3Connection(DQN_AWS_ID, DQN_AWS_SECRET)
    bucket = conn.get_bucket('aiworld')
    episodes = list(bucket.list())
    ret = []
    for episode in episodes:
        if episode.key.find('1414651242') >= 0:  # TODO: Support all batches.
            ret.append(episode)
    return ret


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

    # TODO: Sort by frame.
    votes = fire.get(VOTE_URL + '/' + episode_directory, episode_number)
    # Integrate
    # (image_action, action, game_over, reward, votes)
    if votes:
        votes = votes.values()
        votes = sorted(votes, key=lambda v: (v['frame'], v['subFrame']))
        experiences = combine(votes, episode_data)
        with open(post_filename, 'w', 6) as post_data:
            post_data.write(snappy.compress(json.dumps(experiences)))


def combine(votes, episode_data):
    frames = episode_data['frames']
    i = 0
    while i < len(votes):
        vote = votes[i]
        if vote['session'] == '701aa5f9-9426-4c6f-a082-dfd4ace0c078':
            # Bad session where votes were not recorded for correct episode.
            print 'ignoring bad session vote', vote
            i += 1
            continue
        elif vote['good']:  # Skip good votes for now.
            print 'ignoring', vote['frame'], vote['subFrame']
            i += 1
            continue
        else:
            # Negative rewards are the only crowdsourced rewards we care about
            # for now.
            j = i + 1
            users = {vote['uid']}
            candidate_votes = [vote]
            while j < len(votes):
                next_vote = votes[j]
                if next_vote['frame'] > vote['frame'] + 5:
                    break
                elif not next_vote['good']:
                    # Multiple users need to agree
                    candidate_votes.append(next_vote)
                    users.add(next_vote['uid'])
                j += 1
            i = j
            if len(users) > 1:
                print 'num users: ' + str(len(users))
                median_vote = \
                    candidate_votes[int(round(len(candidate_votes) / 2.0)) - 1]
                sub_frame = \
                    frames[median_vote['frame']][median_vote['subFrame']]
                sub_frame['reward'] = -1
            else:
                # Ignore old negative rewards (game overs)
                try:
                    frames[vote['frame']][vote['subFrame']]['reward'] = 0
                except IndexError, e:
                    print 'vote does not belong to episode!!!', vote

    for i, frame in enumerate(frames):
        for j, sub_frame in enumerate(frame):
            if sub_frame['reward'] != 0:
                print i, j, sub_frame['reward']
    return frames


# def add_votes_property(frames):
#     for frame in frames:
#         for sub_frame in frame:
#             sub_frame['votes'] = 0
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


# def get_all_experience_pairs():
#     filenames = [n for n in os.listdir(INTEGRATE_DIR) if not n.startswith('.')]
#     load_experience_files_in_parallel(filenames)


# def load_experience_files_in_parallel(filenames):
#     ret = []
#     threads = []
#
#     def worker(filename, i):
#         """thread worker function"""
#         print 'loading...' + filename + ' ' + str(i)
#         ret.append(get_experience_pairs(filename))
#         mem_pct = psutil.phymem_usage().percent
#         print 'done with ' + filename + ' ' + str(i) + ' mem ' + str(mem_pct)
#         return
#
#     for i, fn in enumerate(filenames):
#         t = threading.Thread(target=worker, args=(fn, i))
#         threads.append(t)
#         t.start()
#
#     for t in threads:
#         t.join()
#
#     return ret


# def get_experience_pairs(filename):
#     if USE_CACHE and filename in EXPERIENCE_CACHE:
#         print 'EXPERIENCE_CACHE hit!'
#         json_str = read_cache(filename)
#     else:
#         if USE_CACHE:
#             print 'EXPERIENCE_CACHE miss!'
#         with open(INTEGRATE_DIR + filename, 'r') as file_ref:
#             json_str = snappy.decompress(file_ref.read())
#         if USE_CACHE and psutil.phymem_usage().percent < 80:
#             write_cache(filename, json_str)
#         else:
#             if USE_CACHE:
#                 print 'EXPERIENCE_CACHE FULL!'
#     experiences = json.loads(json_str)
#     # compress_screen_hex(experiences)
#     return pair_experiences(experiences, filename)


# def write_cache(filename, json_str):
#     EXPERIENCE_CACHE[filename] = snappy.compress(json_str)
#     print 'EXPERIENCE_CACHE length', len(EXPERIENCE_CACHE)
#
#
# def read_cache(filename):
#     return snappy.decompress(EXPERIENCE_CACHE[filename])
#
#
# def compress_screen_hex(experiences):
#     for e in experiences:
#         for f in e:
#             f['screen_hex'] = snappy.compress(f['screen_hex'])


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


# def test_snappy():
#     # store_integrated_experiences()
#     test = []
#     for i in xrange(35):
#         print i
#         test.append(get_random_experience_pairs())
#     print time.time()
#     for t in test:
#         for p in t:
#             for e in p:
#                 for f in e:
#                     snappy.decompress(f['screen_hex'])


if __name__ == '__main__':
    store_integrated_experiences(parallel=False)
    # get_all_experience_pairs()
