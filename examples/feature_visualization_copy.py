import numpy as np
import matplotlib.pyplot as plt
import caffe

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


MODEL_FILE = 'imagenet/imagenet_deploy.prototxt'
PRETRAINED = 'imagenet/caffe_reference_imagenet_model'

# net = imagenet.ImageNetClassifier(
#     MODEL_FILE, PRETRAINED, center_only=1)

net = caffe.Classifier(caffe_root + 'examples/imagenet/imagenet_deploy.prototxt',
                       caffe_root + 'examples/imagenet/caffe_reference_imagenet_model')
net.set_phase_train()
net.set_mode_cpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_mean('data', caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')  # ImageNet mean
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.set_input_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

scores = net.predict([caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')])

print [(k, v.data.shape) for k, v in net.blobs.items()]

print [(k, v[0].data.shape) for k, v in net.params.items()]

# our network takes BGR images, so we need to switch color channels
def showimage(im):
    if im.ndim == 3:
        im = im[:, :, ::-1]
    plt.imshow(im)
    plt.show()

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

    showimage(data)

# index four is the center crop
image = net.blobs['data'].data[4].copy()
image -= image.min()
image /= image.max()
showimage(image.transpose(1, 2, 0))


# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
