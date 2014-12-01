from boto.s3.connection import S3Connection
from examples.dqn import secrets


def get_episode_list():
    conn = S3Connection(secrets.DQN_AWS_ID, secrets.DQN_AWS_SECRET)
    bucket = conn.get_bucket('aiworld')
    episodes = list(bucket.list())
    return episodes