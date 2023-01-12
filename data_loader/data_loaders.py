import os
import torch
from env.generator import DataGenerator
from torch_geometric.data import InMemoryDataset, Dataset, DataLoader, Data
import numpy as np


class TSPDataLoader:
    def __init__(self, config, batch_size, shuffle=True, validation_split=0.1):
        graph_size = config['arch']['args']['graph_size']
        graph_num = config['data_loader']['data']['graph_num']

        # randomly split the training set and the validation set in proportion
        idx_full = np.arange(graph_num)
        np.random.shuffle(idx_full)
        len_valid = int(graph_num * validation_split)
        valid_idx = idx_full[0: len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        self.dataset_train = DiskDataset(config, 'sample', train_idx, graph_size, graph_num)
        self.dataset_valid = DiskDataset(config, 'sample', valid_idx, graph_size, graph_num, is_valid=True)
        self.dataset_train_IA = DiskDataset(config, 'graph', train_idx, graph_size, graph_num)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def process(self):
        self.data_loader_train = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)
        self.data_loader_valid = DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=self.shuffle)
        self.data_loader_train_IA = DataLoader(self.dataset_train_IA, batch_size=1, shuffle=False)

        return self.data_loader_train, self.data_loader_valid, self.data_loader_train_IA


class DiskDataset(Dataset):
    def __init__(self, config, mode, idx, graph_size, graph_num, is_valid=False, transform=None, pre_transform=None):
        self.mode = mode
        self.idx = idx
        self.config = config
        self.graph_size = graph_size
        self.graph_num = graph_num

        if not is_valid:
            save_dir = os.path.join(config.dataset_processed_dir, 'train', self.mode)
        else:
            save_dir = os.path.join(config.dataset_processed_dir, 'valid')

        super(DiskDataset, self).__init__(save_dir, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        file_names = []
        file_count = 0
        for dir_path, dir_names, filenames in os.walk(self.processed_dir):
            for file in filenames:
                if "data" in file:
                    file_names.append("data_{}.pt".format(file_count))
                    file_count += 1

        if len(file_names) == 0:
            file_names.append("test.pt")

        return file_names

    def __len__(self):
        return len(self.processed_file_names) 

    def download(self):
        pass

    def process(self):
        DataGenerator(self.config, self.mode, self.idx, self.graph_size, self.graph_num).run(self.processed_dir)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data



# class MemoryDataset(InMemoryDataset):
#     def __init__(self, config, transform=None, pre_transform=None):
#         self.config = config
#         save_dir = os.path.join(config['data_loader']['data']['save_dir'],
#                                 config['data_loader']['data']['graph_type'],
#                                 "tsp_{}".format(config['arch']['args']['graph_size']))
#         super(MemoryDataset, self).__init__(save_dir, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def raw_file_names(self):
#         return []
#
#     @property
#     def processed_file_names(self):
#         return ['tsp.dataset']
#
#     def download(self):
#         pass
#
#     def process(self):
#         data_list = DataGenerator(self.config).run()
#
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

