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

def train(solver, net, subsequent_experiences, atari, i):
    experience_one = subsequent_experiences[:EXPERIENCE_SIZE]
    experience_two = subsequent_experiences[EXPERIENCE_SIZE:]
    experience_one_state  = atari.get_state_from_experience (experience_one)
    experience_two_state  = atari.get_state_from_experience (experience_two)
    experience_two_action = atari.get_action_from_experience(experience_two)

    q_values_one = get_q_values(solver, net, experience_one_state)
    q_old = q_values_one[experience_two_action.value]

    q_values_two = get_q_values(solver, net, experience_two_state)
    q_max_index = get_random_q_max_index(q_values_two)
    q_max = q_values_two[q_max_index]

    # TODO: Qnew = q learning update with reward and next_state
    reward = atari.get_reward_from_experience(experience_one)

    # TODO: Set the diff to the q-gradient.
    # net.blobs['fc2'].diff # shape is (1, 4, 1, 1), size = 4
    net.blobs['fc2'].diff[0][0][0][0] = 0.5
    # TODO: Make sure this diff is reflected in fc2's backward C++.
    # Set mutable_cpu_diff of fc2 data to:
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


    # for i, neuron in enumerate(neurons): # Number of actions
    #   bottom_diff[i] = (reward + gamma * q_new


    # Run Backward to update weights.

    # TODO: bprop (Qold - Qnew)^2

    solver.online_update()

    print 'this should be 0.5', net.blobs['fc2'].diff[0][0][0][0]

    # print solver.net.params.values()[0][0].data
    # print [(k, v[0].data.shape) for k, v in solver.net.params.items()]
    if i % 1000 == 0:
        filters = net.params['conv1'][0].data
        vis_square(filters.transpose(0, 2, 3, 1))
    i += 1
#
# I0805 14:08:11.535507 2082779920 inner_product_layer.cpp:100] Craig checking backprop gradient set in python0x1155df000
# I0805 14:08:11.535514 2082779920 inner_product_layer.cpp:82] Craig checking backward inner product
# I0805 14:08:11.537643 2082779920 inner_product_layer.cpp:100] Craig checking backprop gradient set in python0x1176b9e00

# I0805 14:11:21.481334 2082779920 inner_product_layer.cpp:100] Craig checking backprop gradient set in python0x102dabc00
# I0805 14:11:21.481341 2082779920 inner_product_layer.cpp:82] Craig checking backward inner product
# I0805 14:11:21.483536 2082779920 inner_product_layer.cpp:100] Craig checking backprop gradient set in python0x102d51000


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
