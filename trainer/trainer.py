import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils.beam_search import beam_search
from tqdm import tqdm
import math
from numpy import inf
import model.model as module_arch
from torch_geometric.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader_train,
                 data_loader_valid,  data_loader_train_IA, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.data_loader_train_IA = data_loader_train_IA

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader_train)
        else:
            # iteration-based training
            self.data_loader_train = inf_loop(data_loader_train)
            self.len_epoch = len_epoch

        self.do_validation = self.data_loader_valid is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader_train.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)


    def reset(self):
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.start_epoch = 1

        self.model = self.config.init_obj('arch', module_arch)
        self.model = self.model.to(self.device)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.config.init_obj('optimizer', torch.optim, trainable_params)

        self.lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader_train):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            target = data.y.to(self.device)
            loss = self.criterion(output, target)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
       
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return log

    def _test_epoch(self, data_loader, models):
        self.model.eval()
        bs = beam_search(self.config)

        gaps = []
        with torch.no_grad():
           for batch_idx, data in enumerate(data_loader):
                min_gap = inf
                for model in models:
                    tour, l, gap = bs.run(model, data.x.numpy(), data.y.numpy(), self.device)     
                    if gap < min_gap:
                        min_gap = gap  
                gaps.append(min_gap)

        return gaps
    
    def update_data_loader(self, gaps):
        graph_size = self.config['arch']['args']['graph_size']
        #sample_size = int(4 * 2 * graph_size * (graph_size - 1))
        sample_size = graph_size - 1
        weights = [0 for i in range(len(self.data_loader_train.dataset))]
        for i, gap in enumerate(gaps):
            if gap > 0:
                weights[i * sample_size: (i + 1) * sample_size] = [gap] * sample_size

        weights =  np.array(weights) + 1
        #weights = 1 / weights
        weights = torch.DoubleTensor(weights)                                       
        sampler = WeightedRandomSampler(weights, len(weights))
        self.data_loader_train = DataLoader(self.data_loader_train.dataset, batch_size=self.data_loader_train.batch_size, sampler=sampler)        


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        loss_epoch = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.data_loader_valid):
                data = data.to(self.device)
                output = self.model(data) 
                target = data.y.to(self.device)
                l = self.criterion(output, target)
                #ME_loss = []
                #ME_loss.append(l.cpu().numpy())
                #for m in self.trained_models:
                #    o = m(data)
                #    l = self.criterion(o, target)
                #    ME_loss.append(l.cpu().numpy())
                
                #loss_trans = np.array(ME_loss).transpose()
                #min_loss = [np.min(row) for row in loss_trans]
                #loss = np.mean(min_loss)
                loss = l.mean()

                self.writer.set_step((epoch - 1) * len(self.data_loader_valid) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader_train, 'n_samples'):
            current = batch_idx * self.data_loader_train.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
            return base.format(current, total, 100.0 * current / total)

