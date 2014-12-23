import json
import multiprocessing
import random
import itertools
import os.path
from collections import deque
import gc

import numpy as np
import caffe
import matplotlib.pyplot as plt
from PIL import Image
import time
import psutil
import subprocess

from constants import *
import experience_loader
import utils
from ntsc_palette import NTSCPalette
from create_action_sidebar import ActionSidebarImages
import secrets

MAX_HISTORY_LENGTH = 1000000


if INTEGRATE_HUMAN_FEEDBACK:
    experience_pairs = experience_loader.get_queue()
else:
    # This size ends up being cut down considerably due to check_mem,
    # need to implement multiprocessing queue to load from disk as with
    # integrate experiences above.
    experience_pairs = deque(maxlen=MAX_HISTORY_LENGTH)


class Atari(object):
    def __init__(self, log_dir_name, episode_num, start_timestamp, show=False):
        self.experience_pairs = experience_pairs
        self.log_dir_name = log_dir_name
        self.episode_num = episode_num
        self.start_timestamp = start_timestamp
        self.show = show
        self.process = self.launch()
        print 'pid: ', self.process.pid
        self.game_over = False
        self.fin  = open('/s/ale_0.4.4/ale_0_4/ale_fifo_out', 'w+')
        self.fout = open('/s/ale_0.4.4/ale_0_4/ale_fifo_in',  'w+')
        self.i = 0
        self.palette = NTSCPalette()
        self.previous_experience = None
        self.previous_recorded = False
        self.record_rewarding = False
        self.log_file_name = '/frames_' + str(episode_num)
        self.log_file_path = log_dir_name + self.log_file_name
        # Handshake
        self.width, self.height = self.read_width_height()
        self.write('1,0,0,1\n')  # Ask to send (screen, RAM, n/a, episode)
        self.action_images = ActionSidebarImages()
        self.experiences = []
        self.recording   = []

    def launch(self):
        ale_location = "/s/ale_0.4.4/ale_0_4/"
        rom_location = "roms/"
        ale_bin_file = "ale"
        rom_file = 'space_invaders.bin'
        # Run A.L.E
        if self.show:
            show_arg = 'true'
        else:
            show_arg = 'false'
        args = [
            ale_location + ale_bin_file,
            '-run_length_encoding', 'false',
            '-display_screen',      show_arg,
            '-game_controller',     'fifo_named',
            '-frame_skip',          '3',  # TODO: Change to 4 for other games per dqn paper.
            rom_location + rom_file
        ]
        if not INTEGRATE_HUMAN_FEEDBACK:
            self.check_memory()
        return subprocess.Popen(args, cwd='/s/ale_0.4.4/ale_0_4/', close_fds=True)

    # def get_random_feedback(self):
    #     pairs = []
    #     for _ in xrange(25):
    #         pairs += integrate_feedback.get_random_experience_pairs()
    #     ret = deque()
    #     for pair in pairs:
    #         if len(pair) != 2:
    #             print pair
    #         e1, e2 = pair
    #         ret.append([self.deserialize(e1), self.deserialize(e2)])
    #     return ret

    def check_memory(self):
        mem_pct = psutil.phymem_usage().percent
        print 'mem pct is ', mem_pct, 'pairs length', len(self.experience_pairs)
        if mem_pct > 80:
            trim = int(len(self.experience_pairs) / 2.0)
            for _ in xrange(trim):
                try:
                    self.experience_pairs.popleft()
                except Exception, e:
                    import traceback
                    print e
                    print traceback.print_exc()
            gc.collect()

    def add_action_sidebar(self, image, action):
        action_image = self.action_images.images[action.value]
        action_image = np.array(action_image, dtype=np.float64)

        return np.concatenate((image, action_image), axis=1)

    def stop(self):
        utils.close_named_pipe(self.fin)
        utils.close_named_pipe(self.fout)
        pid = self.process.pid
        self.process.kill()  # kill -9 !!!!!!!!!!!!!!!!!!!!
        self.process.wait()
        while utils.check_pid(pid):
            print 'waiting for game to die'
            time.sleep(0.01)  # 10 millis
        if not INTEGRATE_HUMAN_FEEDBACK:
            # We've already recorded this game if we are integrating feedback.
            self.log_frames()  # Keep this at end to avoid stalling atari close.

    def read_width_height(self):
        str_in = self.read()
        while str_in == '':
            time.sleep(1)
            print 'waiting for atari process to start sending data'
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

    def get_random_transition_pairs(self, num):
        if INTEGRATE_HUMAN_FEEDBACK:
            return [experience_pairs.get() for _ in xrange(num)]
        else:
            if len(self.experience_pairs) > num:
                i = random.randint(0, len(self.experience_pairs) - num)
                return list(itertools.islice(self.experience_pairs, i, i + num))
            else:
                return []

    # def record_rewarding_experience(self, experience_pair, total_reward):
    #     if self.previous_experience and total_reward != 0 or self.game_over:
    #         # Record pairs of experiences where the second experience contains
    #         # a reward.
    #         self.rewarding_experience_pairs.append(experience_pair)

    def experience(self, n, action):
        total_reward = 0
        ret = []
        frames = []
        for _ in itertools.repeat(None, n):
            reward, experience, frame = self.experience_frame(action)
            total_reward += reward
            ret.append(experience)
            frames.append(frame)
        if INTEGRATE_HUMAN_FEEDBACK:

            pass
        else:
            # We already stored this experience if we are integrating human feedback.
            self.store_experience(frames, ret)
        return ret

    def store_experience(self, frames, ret):
        self.experiences.append(ret)
        self.recording.append(frames)
        if self.previous_experience:
            experience_pair = (self.previous_experience, ret)
            self.experience_pairs.append(experience_pair)
            # if self.record_rewarding:
            # self.record_rewarding_experience(experience_pair, total_reward)
        self.previous_experience = ret

    def log_frames(self):
        serialized_frames = []
        for frames in self.recording:
            serialized_frames.append(self.serialize(frames))
        with open(self.log_file_path, 'wb') as log_file:
            log_file.write('aiworldFrames = ')
            json.dump({
                'start'   : self.start_timestamp,
                'episode' : self.episode_num,
                'frames'  : serialized_frames
            }, log_file)
        self.zip_and_delete()

    def serialize(self, frames):
        ret = []
        for frame in frames:
            ret.append({
                'image_action' : frame[EXP_IMAGE_ACTION_INDEX].tolist(),
                'action'       : frame[EXP_ACTION_INDEX].name,
                'game_over'    : frame[EXP_GAME_OVER_INDEX],
                'reward'       : frame[EXP_REWARD_INDEX],
                'screen_hex'   : frame[EXP_SCREEN_HEX_INDEX]
            })
        return ret

    def zip_and_delete(self):
        sp = subprocess.Popen([DQN_ROOT + '/zip_and_delete.bash',
                               self.log_dir_name,
                               self.log_file_name,
                               str(self.start_timestamp),
                               str(self.episode_num),
                               secrets.DQN_AWS_ID,
                               secrets.DQN_AWS_SECRET], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = sp.communicate()
        if out:
            print "standard output of subprocess:", out
        if err:
            print "standard error of subprocess:", err
        print "returncode of subprocess:", sp.returncode

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
        frame      = (image_action, action, game_over, reward, screen_hex)
        self.send_action(action)
        return reward, experience, frame

    def get_reward_from_experience(self, experience):
        """Returns sum of rewards
        From DQN paper:
        Since the scale of scores varies greatly from game to game,
        we fixed all positive rewards to be 1 and all negative rewards
        to be  1 leaving 0 rewards unchanged.
        """
        score = sum([e[EXP_REWARD_INDEX] for e in experience])
        ret = 0
        if self.get_game_over_from_experience(experience):
            print '\n\n\n\GAME OVER\n\n\n\n'
            ret = -1
        elif score > 0:
            ret = 1
        elif score < 0:
            ret = -1
        return ret, score

    def get_game_over_from_experience(self, experience):
        return any([e[EXP_GAME_OVER_INDEX] for e in experience])

    def get_state_from_experience(self, experience):
        return [e[EXP_IMAGE_ACTION_INDEX] for e in experience]

    def get_action_from_experience(self, experience):
        return experience[0][EXP_ACTION_INDEX]

    def substitute_reward_in_experience(self, experience, reward):
        k = EXP_REWARD_INDEX
        ret = []
        for f in experience:
            f = list(f)
            x = tuple(f[:k] + [reward] + f[k + 1:])
            ret.append(x)
        return ret

    """ Simulator parsing """

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

        # self.show_image(im)

        im = utils.rgb2gray(im)
        # Resize to dimensions in DQN paper, TODO: pass dims as param.
        im = caffe.io.resize_image_binary(im, (84, 80))
        self.show_checkpoint_image(im)
        if self.show:
            self.im = im
        self.i += 1
        return im


def color_chunks(l):
    """ Yield successive 2-hex chunks from l.
    """
    for i in xrange(0, len(l), 2):
        yield l[i: i + 2]


if __name__ == '__main__':
    print 1
    Popen([DQN_ROOT + '/zip_and_delete.bash', '/Users/cq/Dropbox/src/caffe/examples/dqn/data/episodes/experiences_1411027871', 'experiences_0'])

