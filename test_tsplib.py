import os
import time
import torch
import numpy as np
import argparse
from parse_config import ConfigParser
import model.model as module_arch
import data_loader.data_loaders as module_data
import time 
from utils.beam_search import beam_search
from utils.utils import save_path, load_data, get_batch_for_step
from env.tsp_env import TSPEnv
from torch_geometric.data import DataLoader
import math


def eval(net, config, batch_graphs, selected, test_data, device, graph_size):
    batch, batch_reshape = get_batch_for_step(config, batch_graphs, selected, test_data, device)
    priors = net(batch_reshape, graph_size)

    return priors, batch


def greedy_search(net, config, batch_graphs, device):
    batch_size, graph_size, _ = batch_graphs.shape
    selected = np.zeros((batch_size,), dtype=int)
    test_data = None
    tours = [selected.tolist()]

    for i in range(graph_size - 1):
        action_prob, test_data = eval(net, config, batch_graphs, selected, test_data, device, graph_size)

        action_prob.scatter_(1, torch.tensor(tours).T.to(device), -np.inf)
        action_prob = action_prob.detach().cpu().numpy() 

        selected = np.argmax(action_prob, axis=1)
        tours.append(selected.tolist())

    return np.array(tours).T


def cal_l_g(graph, pred_path, gt_path):
    pred_sorted_locs = graph[np.concatenate((pred_path, [pred_path[0]]))]
    pred_len = np.linalg.norm(pred_sorted_locs[1:] - pred_sorted_locs[:-1], axis=-1).sum()
    gt_sorted_locs = graphs[np.concatenate((gt_path, [gt_path[0]]))]
    gt_len = np.linalg.norm(gt_sorted_locs[1:] - gt_sorted_locs[:-1], axis=-1).sum()
    gap = ((pred_len / gt_len) - 1) * 100
    gap = gap if gap > 0 else 0

    return pred_len, gap

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def test(config):       
    # load data
    test_name = ['berlin52', 'eil51', 'eil76', 'pr76', 'rat99', 'st70', \
                 'a280', 'bier127', 'ch130', 'ch150', 'd198', 'eil101', 'kroA100', 'kroA150', \
                 'kroA200', 'kroB100', 'kroB150', 'kroB200', 'kroC100', 'kroD100', 'kroE100', \
                 'lin105', 'pr107', 'pr124', 'pr136', 'pr144', 'pr152', 'rat195', 'rd100', 'u159', \
                 'gil262', 'lin318', 'pr226', 'pr264', 'pr299', 'ts225', 'tsp225']

    test_graphs = []
    test_gt_paths = []
    for name in test_name:
        test_graphs.append(np.genfromtxt(r'data/test_data/tsplib/{}.tsp'.format(name), dtype=np.float)[:,1:])
        #test_gt_paths.append(np.genfromtxt(r'data/test_data/tsplib/{}.opt.tour'.format(name), dtype=np.int))

    test_gt = [7542, 426, 538, 108159, 1211, 675, \
                    2579, 118282, 6110, 6528, 15780, 629, 21282, 26524, \
                    29368, 22141, 26130, 29437, 20749, 21294, 22068, \
                    14379, 44303, 59030, 96772, 58537, 73682, 2323, 7910, 42080, \
                    2378, 42029, 80369, 49135, 48191, 126643, 3916]
    
    ME_res = []
    for model_id in range(1, 21):
        # load trained model
        model = config.init_obj('arch', module_arch)
        saved_state = torch.load(
                "{}/pretrained_models/model{}_pure_model_best.pth".format(config.save_dir, model_id),
                map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state)
        model = model.to(config['gpu_id'])
        print("Load model successfully ~")

        model.eval()

        res_dict = []
        with torch.no_grad():
            for i in range(len(test_graphs)):
                batch_graphs = normalization(test_graphs[i])
                batch_graphs = torch.tensor(batch_graphs[None,:,:], dtype=torch.float)
                pred_path = greedy_search(model, config, batch_graphs, config['gpu_id'])[0]
            
                pred_sorted_locs = test_graphs[i][np.concatenate((pred_path, [pred_path[0]]))]
                pred_len = np.linalg.norm(pred_sorted_locs[1:] - pred_sorted_locs[:-1], axis=-1).sum()

                gap = (pred_len / test_gt[i] - 1) * 100
                gap = gap if gap > 0 else 0

                res_dict.append(gap)
        ME_res.append(res_dict)

    ME_res_trans = np.array(ME_res).transpose(1,0)
    min_gaps = [np.min(row) for row in ME_res_trans]

    print(min_gaps)
    print(np.mean(min_gaps))
    #print(np.mean(np.array(res_dict).transpose(1,0)[0]))
    #print(np.mean(np.array(res_dict).transpose(1,0)[1]))


    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-i', '--run_id', default=None, type=str,
                      help='')
    args.add_argument('-m', '--model_id', default=None, type=str,
                      help='')
     
    config = ConfigParser.from_args(args, mode='test_divide')

    test(config)
     
