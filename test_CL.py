import numpy as np
import os
from parse_config import ConfigParser
import argparse
from utils.utils import read_path, save_path_gap
import time

args = argparse.ArgumentParser(description='PyTorch Template')
#args.add_argument('-c', '--config', default=None, type=str,
#                  help='config file path (default: None)')
args.add_argument('-i', '--run_id', default=None, type=str,
                  help='')
args.add_argument('-n', '--graph_size', default=None, type=int,
                  help='')
args.add_argument('-nm', '--num_models', default=None, type=int,
                  help='')

config = args.parse_args()

result_dir = os.path.join("saved/SE-GNN", config.run_id, "result/tsp{}/random/greedy".format(config.graph_size))#"result/tsp{}/sampling".format(config.graph_size))
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

num_models = config.num_models 
st = time.time()
ME_lens = []
ME_gaps = []
ME_tours = []
for model_id in range(1, num_models + 1):
    file_path = os.path.join(result_dir, 'model{}_greedy_test_path.txt'.format(model_id))
    lens, gaps, tours = read_path(file_path)
    ME_lens.append(lens)
    ME_gaps.append(gaps)
    ME_tours.append(tours)

gaps_trans = np.array(ME_gaps).transpose()
lens_trans = np.array(ME_lens).transpose()
tours_trans = np.array(ME_tours).transpose((1, 0, 2))

argmin_gaps = [np.argmin(row) for row in gaps_trans]
min_gaps = [gaps_trans[i, argmin_gaps[i]] for i in range(len(argmin_gaps))]
min_lens = [lens_trans[i, argmin_gaps[i]] for i in range(len(argmin_gaps))]
min_tours = [tours_trans[i, argmin_gaps[i]] for i in range(len(argmin_gaps))]


save_path_gap(os.path.join(result_dir, 'ME_greedy_test_path.txt'), \
              min_tours, min_lens, min_gaps)
print("avg len: {} avg gap: {}%".format(np.mean(min_lens), np.mean(min_gaps)))
print("Time consuming to merge: {}".format(time.time() - st))

