from datetime import datetime
import random
import itertools
import os.path
from ntsc_palette import NTSCPalette
import numpy as np
import caffe
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
from collections import namedtuple, deque
from dqn.create_action_sidebar import ActionSidebarImages
import Image

class Atari(object):
    MAX_HISTORY_LENGTH = 1000000

    def __init__(self):
        self.process = self.launch()
        self.game_over = False
        self.fin  = open('/s/ale_0.4.4/ale_0_4/ale_fifo_out', 'w+')
        self.fout = open('/s/ale_0.4.4/ale_0_4/ale_fifo_in',  'w+')
        self.i = 0
        self.palette = NTSCPalette()
        # Handshake
        self.width, self.height = self.read_width_height()
        self.write('1,0,0,1\n')  # Ask to send (Screen, RAM, n/a, Episode)
        self.experience_history = deque(maxlen=self.MAX_HISTORY_LENGTH)
        self.action_images = ActionSidebarImages()

    def stop(self):
        self.fin.close()
        self.fout.close()
        self.process.kill()

    def read_width_height(self):
        str_in = self.read()
        width, height = str_in.split('-')
        print 'width:  ', width
        print 'height: ', height
        return int(width), int(height)

    def read(self):
        return self.fin.readline().strip()

    def write(self, s):
        self.fout.write(s)
        self.fout.flush()

    def show_image(self, im):
        # im = im[:, :, ::-1]
        Image.fromarray(im, 'RGB').save('atari.png')
        plt.imshow(im, interpolation='nearest')
        plt.show()

    def show_checkpoint_image(self, im):
        if self.i % 100 == 0 and os.path.isfile('show_screen'):
            self.show_image(im)

    def send_action(self, action):
        self.write("%d,%d\n" % (action.value, 18))  # 18 = Noop player b.

    def get_random_experience(self, size):
        # Get a random starting frame for the specified window size.
        # So if the window size is four, we don't want 1, 2, or 3 for start.
        # Rather 0, 4, 8, or 12
        max_length = len(self.experience_history)
        if size > max_length:
            return None
        else:
            start = random.randint(0, (max_length - size) / size) * size
            end = start + size
            return list(itertools.islice(self.experience_history, start, end))

    def experience(self, n, action):
        return [self.experience_frame(action) for _ in itertools.repeat(None, n)]
        # images = np.array([images], dtype=np.float32)

    def experience_frame(self, action):
        """ Load frame from game video.
        Returns: (image, action, game_over, reward)
        """
        # screen_hex = width x height x 2-Hex-NTSC-color
        screen_hex, episode, _ = self.read().split(':')
        image = self.get_image_from_screen_hex(screen_hex)
        image_action = self.add_action_sidebar(image, action)
        game_over, reward = self.get_game_over_and_reward(episode)
        experience = (image_action, action, game_over, reward)
        self.experience_history.append(experience)
        self.send_action(action)
        return experience

    def get_reward_from_experience(self, experience):
        """Returns sum of rewards
        """
        total_reward = sum([e[3] for e in experience])
        if any([e[2] for e in experience]):
            # Game over is -100
            print '\n\n\n\n NEGATIVE REWARD FINALLY \n\n\n\n'
            print '\n\n\n\n NEGATIVE REWARD FINALLY \n\n\n\n'
            total_reward = -1

        if total_reward > 0:
            total_reward = 1
        elif total_reward < 0:
            total_reward = -1

        return total_reward

    def get_state_from_experience(self, experience):
        return [e[0] for e in experience]

    def get_action_from_experience(self, experience):
        return experience[0][1]

    def get_game_over_and_reward(self, episode):
        # From ALE manual.pdf:
        # The episode string contains two comma-separated integers
        # indicating episode termination (1 for termination, 0 otherwise)
        # and the most recent reward. It is also colon-terminated.
        game_over, reward = episode.split(',')
        game_over = True if game_over == '1' else False
        reward = int(reward)
        self.game_over = game_over
        return game_over, reward

    def get_image_from_screen_hex(self, screen_hex):
        """ Returns w x h x gray_level """
        colors = []
        for color in color_chunks(screen_hex):
            colors.append(self.palette.colors[int(color, 16)])  # 16 For 2-hex
        # Reshape flat to h x w x RGB
        im = np.reshape(np.array(colors), (self.height, self.width, 3))
        im = rgb2gray(im)
        # Resize to dimensions in DQN paper, TODO: pass dims as param.
        im = caffe.io.resize_image(im, (84, 80))
        self.show_checkpoint_image(im)
        self.i += 1
        if self.i % 10 == 0:
            print datetime.now(), 'ten frames'
        return im

    def launch(self):
        ale_location = "/s/ale_0.4.4/ale_0_4/"
        rom_location = "roms/"
        ale_bin_file = "ale"
        rom_file = 'space_invaders.bin'
        # Run A.L.E
        args = [
            ale_location + ale_bin_file,
            '-run_length_encoding',      'false',
            '-display_screen',           'true',
            '-game_controller',          'fifo_named',
            '-frame_skip',               '3',  # TODO: Change to 4 for other games per dqn paper.
            rom_location + rom_file
        ]
        return Popen(args, cwd='/s/ale_0.4.4/ale_0_4/', close_fds=True)


    def add_action_sidebar(self, image, action):
        action_image = self.action_images.images[action.value]
        action_image = np.array(action_image, dtype=np.float64)

        return np.concatenate((image, action_image), axis=1)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def color_length_chunks(l):
    """ Yield successive 2-hex, 2-hex chunks from l.
    """
    for i in xrange(0, len(l), 4):
        yield l[i: i + 2], l[i + 2: i + 4]


def color_chunks(l):
    """ Yield successive 2-hex, 2-hex chunks from l.
    """
    for i in xrange(0, len(l), 2):
        yield l[i: i + 2]


def repeat(func, times):
    for _ in itertools.repeat(None, times):
        func()
