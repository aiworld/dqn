import numpy as np
import matplotlib.pyplot as plt
import caffe
import skimage.io
import skimage.transform
import sys
from atari import Atari
import dqn.atari_actions as actions
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


def get_best_action(state):
    # fprop the state through the net
    # get the output neuron with the highest activation
    pass

def go():
    setup_matplotlib()
    solver = get_solver()
    net = solver.net
    atari = Atari(show=False)
    i = 0
    action = actions.MOVE_RIGHT
    while True:
        state = atari.get_experience_window(EXPERIENCE_SIZE, action)

        # Dummy values. Just humoring set_input_arrays for now.
        # TODO: Change these to rewards.
        labels = np.array([3.0], dtype=np.float32)

        # [(1, 4, 84, 84)] image stack array (4, exp).
        # Treating frames as channels.
        images = np.array([[frame[0] for frame in state]], dtype=np.float32)

        net.set_input_arrays(images, labels)
        net.forward()

        # TODO: fprop for best action, set Qold
        # TODO: Qnew = q learning update with reward
        # TODO: bprop (Qold - Qnew)^2

        # TODO: Integrate Q-learning algo.
        action = get_best_action(state)  # max Q

        # Train on random action
        data = atari.get_random_experience(EXPERIENCE_SIZE)

        solver.online_update()
        # print solver.net.params.values()[0][0].data
        # print [(k, v[0].data.shape) for k, v in solver.net.params.items()]
        if i % 1000 == 0:
            filters = net.params['conv1'][0].data
            vis_square(filters.transpose(0, 2, 3, 1))
        i += 1


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
