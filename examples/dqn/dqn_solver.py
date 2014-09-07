import random
import math
import os.path
from scipy.spatial import distance
import atari_actions as actions
from utils import *
from constants import LAYER_NAMES
from episode_stats import EpisodeStat
# Epsilon annealed linearly from 1 to 0.1 over the first million frames,
# and fixed at 0.1 thereafter
EPSILON_START = 1.0
EPSILON_ANNEALING_TIME = 1E6
EPSILON_END = 0.1
EPSILON_SLOPE = -(1.0 - EPSILON_END) / EPSILON_ANNEALING_TIME
GAMMA = 0.8  # Using pac-man value.
MINIBATCH_SIZE = 32
LEARNING_RATE = 0.6


class DqnSolver(object):
    def __init__(self, atari, net, solver):
        self.atari           = atari
        self.net             = net
        self.solver          = solver
        self.iter            = 0
        self._forced_exploit = False

    def learn_from_experience_replay(self):
        transition_minibatch = \
            self.atari.get_random_transitions(num=MINIBATCH_SIZE)
        if transition_minibatch:
            transition_minibatch = \
                self.propagate_game_over_backwards(transition_minibatch)
            return self.process_minibatch(transition_minibatch)
        else:
            return EpisodeStat(0.0, [], 0.0)

    def propagate_game_over_backwards(self, transition_minibatch):
        ret = []
        tm = transition_minibatch
        # TODO: Make sure this is not permanently stored.
        for i, transition in enumerate(tm):
            steps_back = 10
            if i > steps_back and \
                    self.atari.get_game_over_from_experience(transition[0]):
                start = max(0, i - steps_back)
                for exp1, exp2 in tm[start:i + 1]:
                    ret.append((
                        self.atari.substitute_reward_in_experience(exp1, -1),
                        self.atari.substitute_reward_in_experience(exp2, -1)
                    ))
                # Assume max of one game over per minibatch
                return tm[:start] + ret + tm[i + 1:]
        return transition_minibatch

    def forward_batch(self, transition_minibatch):
        q_gradients = [0.0] * len(actions.ALL)
        q_sum = 0
        q_max_sum = 0
        q_olds = []
        for transition in transition_minibatch:
            q_max, q_values, action_index, reward = \
                self.get_update_variables(transition)
            q_sum += sum(q_values)
            q_max_sum += q_max
            q_olds.append(q_values)
            q_old = float(q_values[action_index])
            q_new = LEARNING_RATE * (reward + GAMMA * q_max)
            q_gradients[action_index] += q_old - q_new
        return q_gradients, q_max_sum, q_sum, q_olds

    def forward_check(self, q_olds, transition_minibatch):
        """Sanity check that we are moving in the right direction"""
        improvement = 0
        for i, transition in enumerate(transition_minibatch):
            q_max, q_values, action_index, reward = \
                self.get_update_variables(transition)
            q_values_old = q_olds[i]
            for j, q_old in enumerate(q_values_old):
                q_new = LEARNING_RATE * (reward + GAMMA * q_max)
                # assert(q_old < 1)
                improvement += (q_new - q_old)
        return improvement

    def improvement_check_one(self, transition_minibatch):
        for transition in transition_minibatch:
            reward, _ = self.atari.get_reward_from_experience(transition[1])
            if reward != 0:
                q_gradients = [0.0] * len(actions.ALL)
                q_max, q_values, action_index, reward = \
                    self.get_update_variables(transition)
                q_old = float(q_values[action_index])  # copy
                # TODO(Sync with Caffe and momentum)
                q_new = LEARNING_RATE * (reward + GAMMA * q_max)
                q_gradients[action_index] += q_old - q_new
                self.set_gradients_on_caffe_net(q_gradients)
                self.solver.online_update()  # backprop
                _, q_values_updated, _, _ = self.get_update_variables(transition)
                improvement = q_values_updated[action_index] - q_old
                if improvement > 0:
                    pass

    def process_minibatch(self, transition_batch):
        # self.improvement_check_one(transition_batch)
        q_gradients, q_max_sum_orig, q_sum_orig, q_olds = \
            self.forward_batch(transition_batch)
        print 'q_sum_orig: ', q_sum_orig
        q_gradients = 1.0 / float(MINIBATCH_SIZE) * np.array(q_gradients)  # avg
        # TODO: Figure out if loss (not just gradient) needs to be calculated.
        self.set_gradients_on_caffe_net(q_gradients)
        layers_orig = self.get_layer_state()
        self.solver.online_update()  # backprop
        layers_after = self.get_layer_state()
        layer_distances = self.get_layer_distances(layers_orig, layers_after)
        improvement = self.forward_check(q_olds, transition_batch)
        self.show_graphs()
        return EpisodeStat(improvement, layer_distances,
                           l1_norm(q_gradients))

    def show_graphs(self):
        if os.path.isfile('show_graphs'):
            filters = self.net.params['conv1'][0].data
            vis_square(filters.transpose(0, 2, 3, 1))
            filters = self.net.params['conv2'][0].data
            vis_square(filters.reshape(32 *   # Filters
                                       16,    # Dimensions
                                       4, 4)) # h, w
            self.plot_layers()

    def get_layer_state(self):
        ret = []
        for layer_name in LAYER_NAMES:
            ret.append((layer_name, np.copy(self.net.params[layer_name][0].data.flat)))
        return ret

    def get_layer_distances(self, layers_orig, layers_after):
        ret = []
        for i in xrange(len(layers_orig)):
            dist = distance.euclidean(layers_orig[i][1],
                                      layers_after[i][1])
            layer_name = layers_orig[i][0]
            print layer_name, 'distance: ', dist
            ret.append((layer_name, dist))
        return ret

    def set_gradients_on_caffe_net(self, q_gradients):
        # Set mutable_cpu_diff of fc2 data to:
        net = self.net
        for i in xrange(len(actions.ALL)):
            # TODO: May need to reverse this gradient.
            net.blobs['fc2'].diff[0][i] = np.reshape(q_gradients[i], (1, 1, 1))

    def should_exploit(self):
        i = self.iter
        if self.forced_exploit():
            print 'forced exploiting'
            ret = True
        elif i < EPSILON_ANNEALING_TIME:
            ret = random.random() > (float(i) * EPSILON_SLOPE + EPSILON_START)  # 1 to 0.1
            print 'exploiting' if ret else 'exploring'
        else:
            ret = random.random() > EPSILON_END
        return ret

    def forced_exploit(self):
        """Allows manually triggering exploit"""
        global _forced_exploit
        if self.iter % 100 == 0:
            _forced_exploit = os.path.isfile('exploit')
        return _forced_exploit

    def perceive(self, experience):
        atari = self.atari
        state = atari.get_state_from_experience(experience)
        q_values = self.get_q_values(state)
        action_index = self.get_random_q_max_index(q_values)
        return q_values[action_index], actions.ALL.values()[action_index]

    def record_episode_stats(self, episode_stats, experience, q, action,
                             exploit, episode_stat):
        reward, score = self.atari.get_reward_from_experience(experience)
        episode_stats.add(q, reward, score, exploit, action, episode_stat)

    def get_random_q_max_index(self, q_values):
        # Randomly break ties between actions with the same Q (quality).
        max_q = np.nanmax(q_values)
        if math.isnan(max_q):
            raise Exception('Exploding activity values?')
        index_values = filter(lambda x: x[1] == max_q, list(enumerate(q_values)))
        random.shuffle(index_values)
        random_max_index = index_values[0][0]
        return random_max_index

    def get_update_variables(self, experience_pair):
        atari = self.atari
        exp1 = experience_pair[0]
        exp2 = experience_pair[1]
        exp1_state = atari.get_state_from_experience(exp1)
        exp2_state = atari.get_state_from_experience(exp2)
        exp2_action = atari.get_action_from_experience(exp2)
        q_values_one = self.get_q_values(exp1_state)
        print 'q values one', q_values_one
        q_old_action = q_values_one[exp2_action.index]
        print 'old action', q_old_action
        q_values_two = self.get_q_values(exp2_state)
        print 'q_values_two', q_values_two
        q_max_index = self.get_random_q_max_index(q_values_two)
        q_max = q_values_two[q_max_index]
        reward, _ = atari.get_reward_from_experience(exp2)
        # state(exp1) -> action(exp2) -> reward(exp2)
        return q_max, q_values_one, exp2_action.index, reward

    def plot_layers(self):
        net = self.net
        metrics = ['data', 'params', 'gradients']
        data_funcs = {
            'data':      lambda layer: net.blobs[layer].data[0],
            'params':    lambda layer: net.params[layer][0].data,
            'gradients': lambda layer: net.blobs[layer].diff[0]
        }
        num_layers = len(LAYER_NAMES)
        num_metrics = len(metrics)
        f, axarr = plt.subplots(
            num_layers * num_metrics,
            2  # Values and histogram
        )
        i = 0
        for layer_name in LAYER_NAMES:
            for metric in metrics:
                feat = data_funcs[metric](layer_name).flat
                axarr[i, 0].plot(feat)
                axarr[i, 0].set_title(layer_name + ' ' + metric)
                # noinspection PyBroadException
                try:
                    axarr[i, 1].hist(feat[feat > 0], bins=100)
                except:
                    print 'problem with histogram', layer_name, metric
                else:
                    axarr[i, 1].set_title(layer_name + ' ' + metric + ' histogram')
                i += 1
        f.subplots_adjust(hspace=1.3)
        plt.show()

    def get_q_values(self, state):
        """ fprop the state through the net
        get the output neuron with the highest activation"""

        # Dummy values. Just humoring set_input_arrays for now.
        labels = np.array([3.0], dtype=np.float32)

        # [(1, 4, 84, 84)] image stack array (4, exp).
        # Treating frames as channels.
        self.net.set_input_arrays(np.array([state], dtype=np.float32), labels)

        # TODO: Create Q-loss layer in C++ if this is slow.
        self.solver.online_forward()

        # Get top data.
        return list(self.net.blobs['fc2'].data.flat)
        # feat = net.blobs['fc2'].data[4]
        # plt.subplot(2, 1, 1)
        # plt.plot(feat.flat)
        # plt.subplot(2, 1, 2)
        # _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
