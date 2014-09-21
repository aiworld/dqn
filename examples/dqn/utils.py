import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import caffe
import skimage.io
import skimage.transform


# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
import sys
import time
from constants import CAFFE_ROOT, DQN_ROOT


def vis_square(data, im_name, batch, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    save_image(data, im_name, batch)


def get_image_path(im_name, batch):
    path = '%s/data/images_%s/' % (DQN_ROOT, batch)
    if not os.path.exists(path):
        os.makedirs(path)
    return '%s/%s_%d.jpg' % (path, im_name, int(time.time()))


# our network takes BGR images, so we need to switch color channels
def save_image(im, im_name, batch):
    if im.ndim == 3:
        im = im[:, :, ::-1]
    plt.imsave(get_image_path(im_name, batch), im)
    plt.close()
    # plt.show()


def load_stacked_frames_from_disk(filename):
    im = skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)
    im = rgb2gray(im)
    im = caffe.io.resize_image_binary(im, (84, 84))

    if False:
        plt.imshow(im, cmap = plt.get_cmap('gray'))
        plt.show()

    ret = np.array([im, im, im, im], dtype=np.float32)  # Treat frames as channels.

    return ret


def setup_matplotlib():
    # pd.options.display.mpl_style = 'default'
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'


def get_solver(solver_filename):
    sys.path.insert(0, CAFFE_ROOT + 'python')
    solver_file = CAFFE_ROOT + 'examples/dqn/data/solver/dqn_solver.prototxt'
    solver = caffe.SGDSolver(solver_file)
    if solver_filename:
        solver.online_update_setup_resume(solver_filename)
    else:
        solver.online_update_setup()
    return solver


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def l1_norm(v):
    return np.linalg.norm(v, ord=1)


def repeat(func, times):
    for _ in itertools.repeat(None, times):
        func()


def check_pid(pid):
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def close_named_pipe(pipe):
    pipe.flush()
    pipe.close()
    while not pipe.closed:
        time.sleep(0.01)  # 10 millis
