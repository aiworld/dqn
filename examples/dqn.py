import numpy as np
import matplotlib.pyplot as plt
import caffe
import skimage.io
import skimage.transform

def go():
    caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
    import sys
    sys.path.insert(0, caffe_root + 'python')

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'


    # MODEL_FILE = 'imagenet/imagenet_deploy.prototxt'
    MODEL_FILE = caffe_root + 'examples/dqn/dqn_train.prototxt'
    SOLVER_FILE = caffe_root + 'examples/dqn/dqn_solver.prototxt'

    solver = caffe.SGDSolver(SOLVER_FILE)

    data = load_stacked_frames(caffe_root + 'examples/images/cat.jpg')
    data = np.array([data])
    data = np.reshape(data, (1, 4, 84, 84)).astype(np.float32, copy=False)

    # Grayscale four images
    # Turn them into dimensions of input to net.

    labels = np.array([3.0], dtype=np.float32)
    solver.net.set_input_arrays(data, labels)

    solver.online_update_setup()
    solver.online_update()

    for i in xrange(1000):
        solver.online_update()


def load_stacked_frames(filename):
    im = skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)
    im = rgb2gray(im)
    im = caffe.io.resize_image(im, (84, 84))

    if False:
        plt.imshow(im, cmap = plt.get_cmap('gray'))
        plt.show()

    ret = np.array([im, im, im, im], dtype=np.float32)  # Treat frames as channels.

    return ret

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


if __name__ == '__main__':
    go()
