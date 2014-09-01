import time
import utils
from atari import Atari
import atari_actions as actions
from episode_stats import EpisodeStats
from dqn_solver import DqnSolver
from constants import *

EXPERIENCE_WINDOW_SIZE = 4


def go():
    log_file_name = get_episode_log_filename()
    utils.setup_matplotlib()
    solver = utils.get_solver()
    net = solver.net
    atari = Atari()
    episode_count = 0
    action = actions.MOVE_RIGHT_AND_FIRE
    episode_stats = EpisodeStats()
    dqn = DqnSolver(atari, net, solver)
    for i in xrange(int(1E7)):  # 10 million training steps
        experience = atari.experience(EXPERIENCE_WINDOW_SIZE, action)
        q, action = dqn.perceive(experience)
        exploit = dqn.should_exploit()
        if not exploit:
            action = actions.get_random_action()
        episode_stat = dqn.learn_from_experience_replay()
        dqn.record_episode_stats(episode_stats, experience, q, action, exploit,
                                 episode_stat)
        if atari.game_over:
            EpisodeStats.log_csv(episode_count, episode_stats, log_file_name)
            episode_count += 1
            episode_stats = EpisodeStats()
            atari.stop()
            atari = Atari()
        dqn.iter = i


def get_episode_log_filename():
    return '%s/data/episodes/episode_log_%d.csv' % (DQN_ROOT, int(time.time()))

if __name__ == '__main__':
    go()


# def set_gradients(i, net, q_max, q_values, action_index, reward):
#     # NOTE: Q-learning alpha is achieved via neural net (caffe) learning rate.
#     #   (r + gamma * maxQ(s', a') - Q(s, a)) * Q(s, a)
#     #   (r + gamma * q_new - q_old) * q_old
#     # i.e.
#     # q_new = [2, 4, 6, 8]
#     # q_old = [1, 2, 1, 2]
#     # gamma = 0.5
#     # reward = reward[random_state_index] = 2
#     # gamma * q_new = [1, 2, 3, 4]
#     # r + gamma * q_new = [3, 4, 5, 6] # Do this first because it's new value in essence
#     # r + gamma * q_new - q_old = [3, 4, 5, 6] - [1, 2, 1, 2] = [2, 2, 4, 4]
#     # (r + gamma * q_new - q_old) * q_old = [2, 2, 4, 4] * [1, 2, 1, 2] = [2, 4, 4, 8] # Do separately for each neuron / action (not a dot product)
#     # DOES NOT MAKE SENSE THAT BIGGER Q_OLD GIVES BIGGER GRADIENT CRAIG
#     # TODO: Try setting other actions to opposite gradient.
#     q_gradients = [0.0] * len(q_values)
#     q_old = q_values[action_index]
#     q_gradients[action_index] = -(reward + GAMMA * q_max - q_old)  # TODO: Try * q_old to follow dqn paper even though this doesn't make sense as larger q_old should not give larger gradient.
#     print 'reward', reward
#     print 'q_max', q_max
#     set_gradients_on_caffe_net(net, q_gradients)
