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

pass

# deploy for others
# set upload path for user with random key
# pick random or unplayed game for user
# Make votes fuzzy?
# 40 secs
# 25 secs
# 43 secs
# 28
# 59 secs for 1.2MB
# 30 secs for 739k
# 34 777
# 56 1.1MB
# 35 841k
# 1m 1.4M