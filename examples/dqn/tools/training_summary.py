import pandas as pd

data = pd.read_csv('/Users/cq/Downloads/tmp/episode_log_1413430908.csv')
print 'initial 100 game average score:  ', data.score_total[:100].mean()
print 'final   100 game average score:  ', data.score_total[:-100].mean()
print 'initial 100 game average reward: ', data.reward_total[:100].mean()
print 'final   100 game average reward: ', data.reward_total[:-100].mean()

for i in xrange(len(data.score_total) / 100):
    start  = i * 100
    end = start + 100
    print start, '-', end, data.score_total[start:end].mean()