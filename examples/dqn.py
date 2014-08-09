import numpy as np
import matplotlib.pyplot as plt
import caffe
import skimage.io
import skimage.transform
import sys
from atari import Atari
import random
import dqn.atari_actions as actions
import math
from examples.dqn import atari_actions

EXPERIENCE_SIZE = 4


def setup_matplotlib():
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


def get_q_and_best_action(atari, net, experience_window):
    return q, best_action


def go():
    setup_matplotlib()
    solver = get_solver()
    net = solver.net
    atari = Atari(show=False)
    i = 0
    action = actions.MOVE_RIGHT
    while True:
        # TODO: Set skip frame appropriately (4 except for space invaders, 3).
        atari.experience(EXPERIENCE_SIZE, action)
        learn_from_experience_replay(atari, i, net, solver)
        i += 1
        # TODO: Get next action


def learn_from_experience_replay(atari, i, net, solver):
    subsequent_experiences = atari.get_random_experience(EXPERIENCE_SIZE * 2)
    if subsequent_experiences:
        train(solver, net, subsequent_experiences, atari, i)


def get_random_q_max_index(q_values):
    # Randomly break ties between actions with the same Q (quality).
    max_q = np.nanmax(q_values)
    if math.isnan(max_q):
        raise Exception('Exploding activity values?')
    index_values = filter(lambda x: x[1] == max_q, list(enumerate(q_values)))
    random.shuffle(index_values)
    random_max_index = index_values[0][0]
    return random_max_index


def get_gamma(i):
    return 0.8  # Using pacman assignment value.

def train(solver, net, subsequent_experiences, atari, i):
    experience_one = subsequent_experiences[:EXPERIENCE_SIZE]
    experience_two = subsequent_experiences[EXPERIENCE_SIZE:]
    experience_one_state  = atari.get_state_from_experience (experience_one)
    experience_two_state  = atari.get_state_from_experience (experience_two)
    experience_two_action = atari.get_action_from_experience(experience_two)

    q_values_one = get_q_values(solver, net, experience_one_state)
    print 'q values one', q_values_one
    q_old_action = q_values_one[experience_two_action.value]
    print 'old action', q_old_action

    q_values_two = get_q_values(solver, net, experience_two_state)
    print 'q_values_two', q_values_two
    q_max_index = get_random_q_max_index(q_values_two)
    q_max = q_values_two[q_max_index]

    reward = atari.get_reward_from_experience(experience_one)

    # NOTE: Q-learning alpha is achieved via neural net (caffe) learning rate.

    #   (r + gamma * maxQ(s', a') - Q(s, a)) * Q(s, a)
    #   (r + gamma * q_new - q_old) * q_old
    # i.e.
    # q_new = [2, 4, 6, 8]
    # q_old = [1, 2, 1, 2]
    # gamma = 0.5
    # reward = reward[random_state_index] = 2
    # gamma * q_new = [1, 2, 3, 4]
    # r + gamma * q_new = [3, 4, 5, 6] # Do this first because it's new value in essence
    # r + gamma * q_new - q_old = [3, 4, 5, 6] - [1, 2, 1, 2] = [2, 2, 4, 4]
    # (r + gamma * q_new - q_old) * q_old = [2, 2, 4, 4] * [1, 2, 1, 2] = [2, 4, 4, 8] # Do separately for each neuron / action (not a dot product)
    # DOES NOT MAKE SENSE THAT BIGGER Q_OLD GIVES BIGGER GRADIENT CRAIG
    q_gradients = []
    print 'reward', reward
    print 'q_max', q_max
    for q_old in q_values_one:
        q_gradient = (reward + get_gamma(i) * q_max - q_old)  # TODO: Try * q_old to follow dqn paper even though this doesn't make sense as larger q_old should not give larger gradient.
        q_gradients.append(q_gradient)

    # Set mutable_cpu_diff of fc2 data to:
    for i in xrange(len(atari_actions.ALL)):
        net.blobs['fc2'].diff[0][i] = np.reshape( -1.0 * q_gradients[i], (1, 1, 1))  # TODO: May need to reverse this gradient.

    # TODO: Figure out why q_max is exploding to 1.735e+07
    # TODO: Figure out if loss needs to be calculated.
    # TODO: Set the frame skipping.

    print 'before: gradient conv1', net.blobs['conv1'].diff
    print 'before: gradient conv2', net.blobs['conv2'].diff
    print 'before: gradient fc1',   net.blobs['fc1'  ].diff
    print 'before: gradient fc2',   net.blobs['fc2'  ].diff

    plot_layers(net)

    solver.online_update()

    print 'after: gradient conv1', net.blobs['conv1'].diff
    print 'after: gradient conv2', net.blobs['conv2'].diff
    print 'after: gradient fc1',   net.blobs['fc1'  ].diff
    print 'after: gradient fc2',   net.blobs['fc2'  ].diff

    plot_layers(net)

    # FC2 params after one pass goes to 900 without bias_term_


    # print solver.net.params.values()[0][0].data
    # print [(k, v[0].data.shape) for k, v in solver.net.params.items()]
    if True or i % 1000 == 0:
        filters = net.params['conv1'][0].data
        vis_square(filters.transpose(0, 2, 3, 1))


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

if __name__ == '__main__':
    go()
