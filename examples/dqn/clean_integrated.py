import json
import snappy
from multiprocessing import Pool
from constants import INTEGRATE_DIR
from os import listdir
from os.path import isfile, join
import atari_actions

IN_PATH  = '/s/caffe/examples/dqn/data/integrated/episodes/'
OUT_PATH = '/s/caffe/examples/dqn/data/integrated/episodes_clean/'
i = 0


def clean_integrated():
    process_pool = Pool(processes=6)
    for filename in [f for f in listdir(IN_PATH) if isfile(join(IN_PATH, f))]:
        process_pool.apply_async(clean_file, (filename,))
        clean_file(filename)


def clean_file(filename):
    global i
    if filename.find('snappy') >= 0:
        with open(INTEGRATE_DIR + filename, 'r') as file_ref:
            json_str = snappy.decompress(file_ref.read())
            frames = json.loads(json_str)
            out_frames = []
            for frame in frames:
                for exp in frame:
                    del exp['image_action']
                    action_number = atari_actions.ALL[exp['action']].value
                    exp['action_number'] = action_number
                    out_frames.append(exp)
            out_filename = \
                OUT_PATH + 'episode_' + str(i).zfill(3) + \
                '.json.snappy'
            with open(out_filename, 'w', 6) as post_data:
                post_data.write(snappy.compress(json.dumps(out_frames)))
            i += 1


if __name__ == '__main__':
    clean_integrated()