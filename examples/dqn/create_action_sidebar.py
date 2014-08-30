# Make it the same width as the stride.

import numpy as np
import matplotlib.pyplot as plt
import atari_actions


rng = np.random.RandomState(23455)


class ActionSidebarImages(object):
    DIMENSIONS = ['up', 'down', 'left', 'right', 'fire']

    def __init__(self,
        width        = 4,
        height       = 84,
        color        = 1,
        background   = 0,
        action_count = 18
    ):
        self.width           = width
        self.height          = height
        self.color           = color
        self.background      = background
        self.action_count    = action_count
        self.random_nums     = self.initialize_random_nums()
        self._rand_num_index = 0
        self.images          = self.get_images()

    def get_images(self):
        images = {}
        base_images = {}

        all_images = None
        sep = np.zeros((self.height, 1), dtype=np.uint8)
        sep.fill(self.color)

        # Get base image for each dimension.
        for d in self.DIMENSIONS:
            base_images[d] = self.get_random_sparse_image()

        # Combine base images so similar actions have similar representations.
        for name, action in atari_actions.ALL.iteritems():
            image = np.zeros((self.height, self.width), dtype=np.uint8)
            for d in self.DIMENSIONS:
                if getattr(action, d):
                    image |= base_images[d]
            if all_images is None:
                all_images = image
            else:
                all_images = np.concatenate((all_images, sep), axis=1)
                all_images = np.concatenate((all_images, image), axis=1)
            print name
            images[action.value] = image

        if False:
            plt.imshow(all_images, cmap=plt.get_cmap('gray'))
            plt.show()
        return images

    def get_random_sparse_image(self):
        probability = 0.03
        print probability
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        return self.sparsify(image, probability)

    def sparsify(self, image, probability):
        for x in xrange(self.width):
            for y in xrange(self.height):
                if self.get_random_num() <= probability:
                    image[y, x] = self.color
        return image

    def get(self, action):
        return self.images[action.value]

    def initialize_random_nums(self):
        # We want random images to be consistent so that the models
        # can be reused in the future. Using the same seed for
        # np.random.RandomState does this without
        # persisting the images.
        pixels_per_image = self.height * self.width
        num_base_images = len(self.DIMENSIONS)
        num_base_pixels = num_base_images * pixels_per_image
        return rng.uniform(size=num_base_pixels)

    def get_random_num(self):
        ret = self.random_nums[self._rand_num_index]
        self._rand_num_index += 1
        return ret


if __name__ == '__main__':
    ActionSidebarImages()
