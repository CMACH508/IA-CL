import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from env.graph import Graph
import torch
import time
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import copy

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
        
        
def load_data(dir, graph_size):
    graphs = []
    if os.path.exists(dir):
        coordinates = np.genfromtxt(dir).reshape(-1, graph_size, 2)
    
    return coordinates
    #for idx, c in enumerate(coordinates):
    #    vertex_coordinate = c[0:graph_size * 2].reshape(graph_size, 2)
    #    g = Graph(graph_size, vertex_coordinate)
    #    g.init()
    #    graphs.append(g)

    #return graphs


def save_path_gap(file_path, tour, path_len, gap):
    file = open(file_path, 'a')

    for i in range(len(tour)):
        for vertex in tour[i]:
            file.write(str(vertex))
            file.write(' ')

        file.write(str(path_len[i]))
        file.write(' ')
        file.write(str(gap[i]))
        file.write('\n')
    file.close()

def read_path(file_path):
    file = open(file_path, 'r')

    gaps = []
    lens = []
    tours = []
    content = file.read()
    content = content.split('\n')[:-1]
    for line in content:
        line_split = line.split(' ')
        gaps.append(float(line_split[-1]))
        lens.append(float(line_split[-2]))
        tours.append([int(c) for c in line_split[:-2]])
        
    return lens, gaps, tours


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def cal_batch_pdist(batch_graphs):
    diff = (batch_graphs[:,:,None,:] - batch_graphs[:,None,:,:])
    return torch.matmul(diff[:,:,:,None,:], diff[:,:,:,:,None]).squeeze(-1).squeeze(-1).sqrt()

def cal_k_neighbor(batch_graphs, k_n):
    batch_size, graph_size, _ = batch_graphs.shape

    dis_mat = cal_batch_pdist(batch_graphs)

    knn_mats, target_ver = torch.topk(dis_mat, k_n + 1, largest=False)
    knn_mats = knn_mats[:,:,1:]
    target_ver = target_ver[:,:,1:]

    source_ver = torch.ones([batch_size * graph_size, k_n], dtype=torch.long) * torch.arange(0, batch_size * graph_size).unsqueeze(1)
    shift_dis = torch.ones([batch_size, k_n * graph_size], dtype=torch.long) * torch.arange(0, batch_size * graph_size, graph_size).unsqueeze(1)
    source_ver = source_ver.to(target_ver.device)
    shift_dis = shift_dis.to(target_ver.device)

    target_ver = target_ver.reshape(-1) + shift_dis.reshape(-1)
    edge_index = torch.stack([source_ver.reshape(-1), target_ver], dim=0)
    
    return edge_index, knn_mats, dis_mat

#def cal_tour_len(batch_graphs, pred_paths, gt_paths):
#    dis_mat = cal_batch_pdist(batch_graphs)
#
#    pred_lens = []
#    gaps = []
#    for i in range(len(dis_mat)):
#        pred_len = dis_mat[i][pred_paths[i], np.roll(pred_paths[i], -1)].sum().item()
#        gt_len = dis_mat[i][gt_paths[i], np.roll(gt_paths[i], -1)].sum().item()
#        pred_lens.append(pred_len)
#        gaps.append(((pred_len / gt_len) - 1) * 100)
#    
#    return pred_lens, gaps

def save_path(file_path, graphs, pred_paths, gt_paths):
    file = open(file_path, 'a')

    for i in range(len(graphs)):
        pred_sorted_locs = graphs[i][np.concatenate((pred_paths[i], [pred_paths[i][0]]))]
        pred_len = np.linalg.norm(pred_sorted_locs[1:] - pred_sorted_locs[:-1], axis=-1).sum()
        gt_sorted_locs = graphs[i][np.concatenate((gt_paths[i], [gt_paths[i][0]]))]
        gt_len = np.linalg.norm(gt_sorted_locs[1:] - gt_sorted_locs[:-1], axis=-1).sum()
        gap = ((pred_len / gt_len) - 1) * 100
        gap = gap if gap > 0 else 0

        for vertex in pred_paths[i]:
            file.write(str(vertex))
            file.write(' ')

        file.write(str(pred_len))
        file.write(' ')
        file.write(str(gap))
        file.write('\n')
    
    file.close()

class TestData:
    edge_attr = torch.tensor([], dtype=torch.float)
    edge_index = torch.tensor([], dtype=torch.long)
    x = torch.tensor([], dtype=torch.float)
    batch = None
    dis_mat = None

def get_batch_for_step(config, batch_graphs, selected, test_data, device):
    #selected = torch.tensor(selected, dtype=torch.long)
    batch_size, graph_size, _ = batch_graphs.shape

    if not test_data:    # special for the first step
        test_data = TestData()
        test_data.x = torch.zeros([batch_size, graph_size, config['arch']['args']['node_dim']], dtype=torch.float).to(device)
        test_data.x[:,:,0].scatter_(1, selected[:,None], 1)
        test_data.x[:,:,1:3] = batch_graphs
        test_data.x[:,:,3] = 1
        test_data.x[:,:,4:6] = batch_graphs[:,0,:].unsqueeze(dim=1)
        
        edge_index, knn_mats, dis_mat = cal_k_neighbor(batch_graphs, config['data_loader']['data']['knn'])
        test_data.edge_attr = torch.zeros([batch_size, graph_size, config['data_loader']['data']['knn'], config['arch']['args']['edge_dim']], dtype=torch.float).to(device)
        test_data.edge_attr[:,:,:,0] = knn_mats
        test_data.edge_attr[:,:,:,1] = 1
        test_data.edge_attr[:,:,:,2:4] = batch_graphs.gather(1, selected[:,None,None].expand(batch_size, graph_size, 2)).unsqueeze(2)
        test_data.edge_index = edge_index.to(device)
        test_data.dis_mat = dis_mat.to(device)

        test_data_reshape = copy.copy(test_data)
        test_data_reshape.x = test_data_reshape.x.reshape(-1, config['arch']['args']['node_dim']).to(device)
        test_data_reshape.edge_attr = test_data_reshape.edge_attr.reshape(-1, config['arch']['args']['edge_dim']).to(device)
        test_data_reshape.edge_index = test_data_reshape.edge_index.reshape(2, -1).to(device)
    
    else:
        test_data.x[:,:,0].scatter_(1, selected[:,None], 1)
        test_data.edge_attr[:,:,:,2:4] = batch_graphs.gather(1, selected[:,None,None].expand(batch_size, graph_size, 2)).unsqueeze(2)
       
        test_data_reshape = copy.copy(test_data)
        test_data_reshape.x = test_data_reshape.x.reshape(-1, config['arch']['args']['node_dim']).to(device)
        test_data_reshape.edge_attr = test_data_reshape.edge_attr.reshape(-1, config['arch']['args']['edge_dim']).to(device)
        test_data_reshape.edge_index = test_data_reshape.edge_index.reshape(2, -1).to(device)

    return test_data, test_data_reshape
