import os
import os.path as osp
import torch
import numpy as np
from env.base_graph import BaseGraph
from env.simulator import TspSimulator
from torch_geometric.data import Data
from tqdm import tqdm


class DataGenerator:
    def __init__(self, config, mode, idx, graph_size, graph_num):
        super(DataGenerator, self).__init__()
        self.mode = mode
        self.idx = idx
        self.config = config
        self.graph_size = graph_size
        self.graph_num = graph_num
        # Data file path 
        self.points_dir = os.path.join('data', 'train_data',
                                        self.config['data_loader']['data']['graph_type'],
                                       'tsp{}'.format(self.config['arch']['args']['graph_size']),
                                       'tsp{}_train.txt'.format(self.config['arch']['args']['graph_size']))
        self.path_dir = os.path.join('data', 'train_data', 
                                       self.config['data_loader']['data']['graph_type'],
                                      'tsp{}'.format(self.config['arch']['args']['graph_size']),
                                      'tsp{}_train_path.txt'.format(self.config['arch']['args']['graph_size']))

    def generate_data(self, graph, path):
        g = BaseGraph(self.graph_size, graph)
        g.init_graph(self.config['data_loader']['data']['knn'])
        path = path.astype(dtype=int).tolist()
        simulator = TspSimulator(self.config, g, path)
        data = simulator.play()

        return data

    def augmentation_path(self, path):
        paths = []
        for i in range(len(path)):
            p = np.concatenate((path[i:], path[0:i]))
            paths.append(p)

        path = path[::-1]
        for i in range(len(path)):
            p = np.concatenate((path[i:], path[0:i]))
            paths.append(p)

        return paths

    def augmentation_graph(self, graph):
        graphs = [graph]

        # Flip horizontal
        graph_h = graph
        graph_h[:,0] = 1 - graph_h[:,0]
        graphs.append(graph_h)

        # Flip vertical
        graph_v = graph
        graph_v[:,1] = 1 - graph_v[:,1]
        graphs.append(graph_v)

        # Flip both
        graph_b = graph
        graph_b = 1 - graph_b
        graphs.append(graph_b)
      
        return graphs

    def augmentation(self, graph, path):
        aug_graphs = []
        aug_paths = []
        
        gs = self.augmentation_graph(graph)
        ps = self.augmentation_path(path)
        
        aug_graphs = np.repeat(gs, len(ps), axis=0)
        aug_paths = ps * 4
            
        return aug_graphs, aug_paths

    def run(self, dir):
        graphs, paths = self.load_data()
        i = 0
        for (graph, path) in tqdm(zip(graphs, paths), total=len(graphs), desc='Generate Data'):
            if self.mode == 'sample':
                # Data augumentation
                #aug_graphs, aug_paths = self.augmentation(graph, path)
                #for (aug_graph, aug_path) in zip(aug_graphs, aug_paths):
                #    data = self.generate_data(aug_graph, aug_path)
                #    for d in data:
                #        torch.save(d, osp.join(dir, 'data_{}.pt'.format(i)))
                #        i += 1
                data = self.generate_data(graph, path)
                for d in data:
                    torch.save(d, osp.join(dir, 'data_{}.pt'.format(i)))
                    i += 1
            else:
                graph = torch.tensor(graph, dtype=torch.float)
                path = torch.tensor(path, dtype=torch.long)
                data = Data(x=graph, y=path)
                torch.save(data, osp.join(dir, 'data_{}.pt'.format(i)))
                i += 1
 
    def load_data(self):
        graphs = np.genfromtxt(self.points_dir, dtype=np.float).reshape(-1, self.graph_size, 2)[self.idx, :, :]
        paths = np.genfromtxt(self.path_dir)[self.idx, :]

        return graphs, paths
