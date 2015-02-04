import pandas as pd
import matplotlib.pyplot as plt
from dateutil.parser import parse as dateparse
from dateutil import tz
import grab_training_file_from_dropbox
import arrow

print 'downloading csv...'
grab_training_file_from_dropbox.get_latest()
data = pd.read_csv('latest_log.csv')

max_mean = 0
max_info = None
means = []
end = 0
for i in xrange(len(data.score_total) / 100):
    start = i * 100
    end = min(start + 100, len(data.score_total))
    mean = data.score_total[start:end].mean()
    means.append(mean)
    if max_mean < mean:
        max_mean = mean
        max_info = (mean, start, end)

exploits          = data.num_exploits[max_info[2]]
game_moves        = data.length[max_info[2]]
exploits_latest   = data.num_exploits[end]
game_moves_latest = data.length[end]

game_duration = (dateparse(data.time[end]) - dateparse(data.time[end - 1]))\
    .total_seconds()
last_game_time = dateparse(list(data.time)[-1])\
    .replace(tzinfo=tz.tzutc())\
    .astimezone(tz.tzlocal())
print 'max avg score:', (str(max_info[0]) + ','),\
    'starting at:', max_info[1], '/', len(data.score_total),\
    'with exploitation of',\
    str(int(round(float(exploits) / game_moves * 100.0))) + '%',\
    '(' + str(exploits) + '/' + str(game_moves) + ')',\
    '- latest exploitation',\
    str(int(round(float(exploits_latest) / game_moves_latest * 100.0))) + '%',\
    '(' + str(exploits_latest) + '/' + str(game_moves_latest) + ')',\
    '-', str(float(game_moves_latest) / game_duration),  \
    '(' + str(game_moves_latest) + '/' + str(int(round(game_duration))) + ')', \
    'moves/second -', arrow.get(last_game_time).humanize()

ts = pd.Series(means)
ts.plot()
plt.savefig('average_scores.png')

# print 'initial 100 game average score:  ', data.score_total[:100].mean()
# print 'final   100 game average score:  ', data.score_total[:-100].mean()
# print 'initial 100 game average reward: ', data.reward_total[:100].mean()
# print 'final   100 game average reward: ', data.reward_total[:-100].mean()