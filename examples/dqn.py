import numpy as np
import matplotlib.pyplot as plt
import caffe
import skimage.io
import skimage.transform
import time
from datetime import datetime
import sys
from atari import Atari
import random
import dqn.atari_actions as actions
import math
from dqn import atari_actions
import os.path
import pandas as pd
from dqn.episode_stats import EpisodeStats

EXPERIENCE_WINDOW_SIZE = 4

# Epsilon annealed linearly from 1 to 0.1 over the first million frames,
# and fixed at 0.1 thereafter
EPSILON_START = 1.0
EPSILON_ANNEALING_TIME = 1E6
EPSILON_END = 0.1
EPSILON_SLOPE = -(1.0 - EPSILON_END) / EPSILON_ANNEALING_TIME

GAMMA = 0.8  # Using pac-man value.

MINIBATCH_SIZE = 32

def setup_matplotlib():
    # pd.options.display.mpl_style = 'default'
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'


def get_solver():
    caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
    sys.path.insert(0, caffe_root + 'python')
    solver_file = caffe_root + 'examples/dqn/dqn_solver.prototxt'
    solver = caffe.SGDSolver(solver_file)
    solver.online_update_setup()
    return solver


def go():
    setup_matplotlib()
    solver = get_solver()
    net = solver.net
    atari = Atari()
    episode_count = 0
    action = actions.MOVE_RIGHT
    episode_stats = EpisodeStats()
    for i in xrange(int(1E7)):  # 10 million training steps
        experience = atari.experience(EXPERIENCE_WINDOW_SIZE, action)
        q, action = perceive(atari, solver, net, experience)
        exploit = should_exploit(i)
        if not exploit:
            action = atari_actions.get_random_action()
        record_episode_stats(episode_stats, atari, experience, q, action, exploit)
        learn_from_experience_replay(atari, i, net, solver)
        if atari.game_over:
            log_csv(episode_count, episode_stats)
            episode_count += 1
            episode_stats = EpisodeStats()
            atari.stop()
            atari = Atari()


def learn_from_experience_replay(atari, i, net, solver):
    transition_minibatch = \
        atari.get_random_transitions(num=MINIBATCH_SIZE)
    q_gradients = [0.0] * len(atari_actions.ALL)
    q_sum_orig = 0

    for transition in transition_minibatch:
        q_max, q_values, action_index, reward = \
            get_update_variables(atari, net, solver, transition)
        q_sum_orig += sum(q_values)
        for j, q_old in enumerate(q_values):
            q_new = reward + GAMMA * q_max
            # assert(q_old < 1)
            q_gradients[j] += q_old - q_new
    print 'q_sum_orig: ', q_sum_orig
    q_sum_after = 0
    if transition_minibatch:
        q_gradients = 1.0 / float(MINIBATCH_SIZE) * np.array(q_gradients)  # avg
        # TODO: Figure out if loss (not just gradient) needs to be calculated.
        set_gradients_on_caffe_net(net, q_gradients)
        solver.online_update()
        for transition in transition_minibatch:
            q_max, q_values, action_index, reward = \
                get_update_variables(atari, net, solver, transition)
            q_sum_after += sum(q_values)
    print 'q_sum_after: ', q_sum_after
    print 'q diff: ', (q_sum_after - q_sum_orig)


    if os.path.isfile('show_graphs'):
        filters = net.params['conv1'][0].data
        vis_square(filters.transpose(0, 2, 3, 1))
        plot_layers(net)


def set_gradients_on_caffe_net(net, q_gradients):
    # Set mutable_cpu_diff of fc2 data to:
    for i in xrange(len(atari_actions.ALL)):
        # TODO: May need to reverse this gradient.
        net.blobs['fc2'].diff[0][i] = np.reshape(q_gradients[i], (1, 1, 1))


def should_exploit(i):
    if forced_exploit(i):
        print 'forced exploiting'
        ret = True
    elif i < EPSILON_ANNEALING_TIME:
        ret = random.random() > (float(i) * EPSILON_SLOPE + EPSILON_START)  # 1 to 0.1
        print 'exploiting' if ret else 'exploring'
    else:
        ret = random.random() > EPSILON_END
    return ret


_forced_exploit = False


def forced_exploit(i):
    """Allows manually triggering exploit"""
    global _forced_exploit
    if i % 100 == 0:
        _forced_exploit = os.path.isfile('exploit')
    return _forced_exploit


def perceive(atari, solver, net, experience):
    state = atari.get_state_from_experience(experience)
    q_values = get_q_values(solver, net, state)
    action_index = get_random_q_max_index(q_values)
    return q_values[action_index], atari_actions.ALL.values()[action_index]


def record_episode_stats(episode_stats, atari, experience, q, action, exploit):
    reward = atari.get_reward_from_experience(experience)
    episode_stats.add(q, reward, exploit, action)


def get_random_q_max_index(q_values):
    # Randomly break ties between actions with the same Q (quality).
    max_q = np.nanmax(q_values)
    if math.isnan(max_q):
        raise Exception('Exploding activity values?')
    index_values = filter(lambda x: x[1] == max_q, list(enumerate(q_values)))
    random.shuffle(index_values)
    random_max_index = index_values[0][0]
    return random_max_index


def get_update_variables(atari, net, solver, experience_pair):
    # TODO: Take out actions that don't matter for Space Invaders (up, down)
    experience_one = experience_pair[0]
    experience_two = experience_pair[1]
    experience_one_state = atari.get_state_from_experience(experience_one)
    experience_two_state = atari.get_state_from_experience(experience_two)
    experience_two_action = atari.get_action_from_experience(experience_two)
    q_values_one = get_q_values(solver, net, experience_one_state)
    print 'q values one', q_values_one
    q_old_action = q_values_one[experience_two_action.value]
    print 'old action', q_old_action
    q_values_two = get_q_values(solver, net, experience_two_state)
    print 'q_values_two', q_values_two
    q_max_index = get_random_q_max_index(q_values_two)
    q_max = q_values_two[q_max_index]
    reward = atari.get_reward_from_experience(experience_two)
    return q_max, q_values_one, experience_two_action.value, reward


def plot_layers(net):
    names = ['conv1', 'conv2', 'fc1', 'fc2']
    metrics = ['data', 'params', 'gradients']
    data_funcs = {
        'data':      lambda net, name: net.blobs[name].data[0],
        'params':    lambda net, name: net.params[name][0].data,
        'gradients': lambda net, name: net.blobs[name].diff[0]
    }
    num_layers = len(names)
    num_metrics = len(metrics)
    f, axarr = plt.subplots(
        num_layers * num_metrics,
        2  # Values and histogram
    )
    i = 0
    for name in names:
        for metric in metrics:
            feat = data_funcs[metric](net, name).flat
            axarr[i, 0].plot(feat)
            axarr[i, 0].set_title(name + ' ' + metric)

            try:
                axarr[i, 1].hist(feat[feat > 0], bins=100)
            except:
                print 'problem with histogram', name, metric
            else:
                axarr[i, 1].set_title(name + ' ' + metric + ' histogram')
            i += 1
    f.subplots_adjust(hspace=1.3)
    plt.show()


def get_q_values(solver, net, state):
    """ fprop the state through the net
    get the output neuron with the highest activation"""

    # Dummy values. Just humoring set_input_arrays for now.
    labels = np.array([3.0], dtype=np.float32)

    # [(1, 4, 84, 84)] image stack array (4, exp).
    # Treating frames as channels.
    net.set_input_arrays(np.array([state], dtype=np.float32), labels)

    # TODO: Create Q-loss layer in C++ if this is slow.
    solver.online_forward()

    # Get top data.
    return list(net.blobs['fc2'].data.flat)
    # feat = net.blobs['fc2'].data[4]
    # plt.subplot(2, 1, 1)
    # plt.plot(feat.flat)
    # plt.subplot(2, 1, 2)
    # _ = plt.hist(feat.flat[feat.flat > 0], bins=100)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    show_image(data)


# our network takes BGR images, so we need to switch color channels
def show_image(im):
    if im.ndim == 3:
        im = im[:, :, ::-1]
    plt.imshow(im)
    plt.show()


def load_stacked_frames_from_disk(filename):
    im = skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)
    im = rgb2gray(im)
    im = caffe.io.resize_image(im, (84, 84))

    if False:
        plt.imshow(im, cmap = plt.get_cmap('gray'))
        plt.show()

    ret = np.array([im, im, im, im], dtype=np.float32)  # Treat frames as channels.

    return ret


log_file_name = 'episode_log_' + str(int(time.time())) + '.csv'


def log_csv(episode_count, episode_stats):
    aggregates = episode_stats.aggregates()
    with open(log_file_name, 'a') as log:
        if episode_count == 0:
            # Write headers
            log.write(','.join(['time'] + aggregates.keys()) + '\n')
        cols = [datetime.utcnow()] + aggregates.values()
        log.write(','.join([str(c) for c in cols]) + '\n')



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


if __name__ == '__main__':
    go()

