from examples.dqn.constants import EPISODE_LIST_URL, FIREBASE_URL, BATCH_LIST_URL
from examples.dqn.secrets import ADMIN_EMAIL, ADMIN_PASSWORD, FIREBASE_KEY
from firebase import firebase as fb
from examples.dqn.tools import s3
from collections import Counter


def go():
    auth = fb.FirebaseAuthentication(FIREBASE_KEY, ADMIN_EMAIL, ADMIN_PASSWORD)
    fire = fb.FirebaseApplication(FIREBASE_URL, auth)
    s3_episodes = sorted(s3.get_episode_list(), key=lambda ep: sort_key(ep))
    populate_batch_counts(fire, s3_episodes)
    populate_episodes_by_length(fire, s3_episodes)


def populate_batch_counts(fire, s3_episodes):
    batches = []
    for e in s3_episodes:
        batches.append(int(e.name[:e.name.index('/')]))
    batch_counts = Counter(batches)
    for batch, count in batch_counts.iteritems():
        fire.put(BATCH_LIST_URL, batch, count)


def populate_episodes_by_length(fire, s3_episodes):
    episode_list = fire.get(EPISODE_LIST_URL, None) or {}
    index_by_size = 0
    prev_key = None
    for ep in s3_episodes:
        # Store episodes by increasing size.
        batch, index = ep.name.split('/')
        key = int(batch)
        if prev_key != key:
            index_by_size = 0
        else:
            index_by_size += 1
        prev_key = key
        if not episode_list.has_key(key):
            url = '/'.join([EPISODE_LIST_URL, batch])
            result = fire.put(url, str(index_by_size),
                              {'origIndex': int(index), 'size': ep.size})
            print 'added', key, result


def sort_key(ep):
    return (
        int(ep.name.split('/')[0]),  # episode batch start
        ep.size,                     # side in bytes of episode ~length
        int(ep.name.split('/')[1])   # original episode number
    )


if __name__ == '__main__':
    go()