import pandas as pd
from collections import OrderedDict

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

    def add(self, q, reward, exploit, action):
        self.q_values.append(q)
        self.rewards.append(reward)
        self.exploits.append(exploit)
        if exploit:
            self.exploit_qs.append(q)
            self.exploit_rewards.append(reward)
            self.exploit_actions.append(action.value)
        else:
            self.explore_qs.append(q)
            self.explore_rewards.append(reward)

    def aggregates(self):
        q_series               = pd.Series(self.q_values)
        reward_series          = pd.Series(self.rewards)
        exploit_qs_series      = pd.Series(self.exploit_qs)
        explore_qs_series      = pd.Series(self.explore_qs)
        exploit_rewards_series = pd.Series(self.exploit_rewards)
        explore_rewards_series = pd.Series(self.explore_rewards)
        ret = OrderedDict()
        for name, var in locals().iteritems():
            if name.endswith('series'):
                self.add_series_vars(name, var, ret)

        exploit_series = pd.Series(self.exploits)
        ret['num_exploits'] = \
            exploit_series.where(exploit_series == True).count()

        ret['exploit_action_count'] = len(set(self.exploit_actions))
        ret['length'] = len(self.rewards)
        return ret

    def add_series_vars(self, name, series, d):
        stats           = OrderedDict(series.describe())
        stats['median'] = series.median()
        stats['total']  = series.sum()
        stats['range']  = series.max() - series.min()

        name = name.replace('series', '')
        for k, v in stats.iteritems():
            d[name + k] = v
