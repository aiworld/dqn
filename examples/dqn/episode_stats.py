from datetime import datetime
import pandas as pd
from collections import OrderedDict
from constants import LAYER_NAMES


class EpisodeStats(object):
    def __init__(self):
        self.q_values        = []
        self.rewards         = []
        self.exploits        = []
        self.exploit_rewards = []
        self.exploit_qs      = []
        self.explore_rewards = []
        self.explore_qs      = []
        self.exploit_actions = []
        self.improvements    = []
        self.l1_loss         = []
        self.episode_count   = 0
        self.layer_distances = {}
        for layer_name in LAYER_NAMES:
            self.layer_distances[layer_name] = []

    def add(self, q, reward, exploit, action, episode_stat):
        self.q_values           .append(q)
        self.rewards            .append(reward)
        self.exploits           .append(exploit)
        self.improvements       .append(episode_stat.improvement)
        self.l1_loss            .append(episode_stat.l1_loss)
        self.add_layer_distances(episode_stat.layer_distances)
        if exploit:
            self.exploit_qs     .append(q)
            self.exploit_rewards.append(reward)
            self.exploit_actions.append(action.value)
        else:
            self.explore_qs     .append(q)
            self.explore_rewards.append(reward)

    def add_layer_distances(self, layer_distances):
        for dist in layer_distances:
            self.layer_distances[dist[0]].append(dist[1])

    def aggregates(self):
        q_series               = pd.Series(self.q_values)
        reward_series          = pd.Series(self.rewards)
        exploit_qs_series      = pd.Series(self.exploit_qs)
        explore_qs_series      = pd.Series(self.explore_qs)
        exploit_rewards_series = pd.Series(self.exploit_rewards)
        improvements_series    = pd.Series(self.improvements)
        l1_loss_series         = pd.Series(self.l1_loss)
        ret = OrderedDict()
        for name, var in sorted(locals().iteritems()):
            if name.endswith('series'):
                self.add_series_vars(name, var, ret)
        for layer_name in LAYER_NAMES:
            self.add_series_vars(
                layer_name + '_distance_',
                pd.Series(self.layer_distances[layer_name]), ret)
        exploit_series = pd.Series(self.exploits)
        ret['num_exploits'] = \
            exploit_series.where(exploit_series == True).count()
        ret['exploit_action_count'] = len(set(self.exploit_actions))
        ret['length'] = len(self.rewards)
        return ret

    def add_series_vars(self, name, series, d):
        print 'adding series for ' + name
        stats           = OrderedDict(series.describe())
        stats['median'] = series.median()
        stats['total']  = series.sum()
        stats['range']  = series.max() - series.min()

        name = name.replace('series', '')
        for k, v in stats.iteritems():
            d[name + k] = v

    @staticmethod
    def log_csv(episode_count, episode_stats, log_file_name):
        aggregates = episode_stats.aggregates()
        with open(log_file_name, 'a') as log:
            if episode_count == 0:
                # Write headers
                log.write(','.join(['time'] + aggregates.keys()) + '\n')
            cols = [datetime.utcnow()] + aggregates.values()
            log.write(','.join([str(c) for c in cols]) + '\n')


class EpisodeStat():
    def __init__(self, improvement, layer_distances, l1_loss):
        self.improvement     = improvement
        self.layer_distances = layer_distances
        self.l1_loss         = l1_loss
