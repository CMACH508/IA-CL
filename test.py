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



def eval(net, config, batch_graphs, selected, test_data, device):
    batch, batch_reshape = get_batch_for_step(config, batch_graphs, selected, test_data, device)
    priors = net(batch_reshape)

    return priors, batch


def greedy_search(net, config, batch_graphs, device):
    batch_size, graph_size, _ = batch_graphs.shape
    selected = torch.zeros((batch_size,), dtype=int).to(device)
    test_data = None
    tours =  torch.zeros((1, batch_size), dtype=torch.long).to(device)

    for i in range(graph_size - 1):
        action_prob, test_data = eval(net, config, batch_graphs, selected, test_data, device)
        action_prob.scatter_(1, tours.T, -np.inf)
        selected = action_prob.argmax(dim=1)
        tours = torch.cat([tours, selected.unsqueeze(0)])

    return tours.T


def get_edge_index(sampling_size, batch_size, graph_size, k_n, target_ver):
    source_ver = torch.ones([batch_size * sampling_size * graph_size, k_n], dtype=torch.long) \
                 * torch.arange(0, batch_size * sampling_size * graph_size).unsqueeze(1)
    source_ver = source_ver.to(target_ver.device)

    # reduction shift
    shift_dis = torch.ones([batch_size, k_n * graph_size], dtype=torch.long) \
                 * torch.arange(0, batch_size * graph_size, graph_size).unsqueeze(1)
    shift_dis = shift_dis.reshape(-1).to(target_ver.device)
    target_ver = target_ver - shift_dis
    target_ver = target_ver.reshape(batch_size, -1).repeat(1, sampling_size).reshape(-1)

    # Recalculate shift
    shift_dis = torch.ones([batch_size * sampling_size, k_n * graph_size], dtype=torch.long) \
                 * torch.arange(0, batch_size * sampling_size * graph_size, graph_size).unsqueeze(1)
    shift_dis = shift_dis.reshape(-1).to(target_ver.device)


    target_ver = target_ver + shift_dis
    edge_index = torch.stack([source_ver.reshape(-1), target_ver], dim=0)

    return edge_index


def sampling(net, config, batch_graphs, device):
    dis_st = time.time()
    batch_size, graph_size, _ = batch_graphs.shape
    selected = torch.zeros((batch_size,), dtype=int).to(device)
    test_data = None
    sampling_size = config['sampling']['size']
    tours = torch.zeros((1, batch_size * sampling_size), dtype=torch.long).to(device)
    lens = torch.zeros((batch_size * sampling_size), dtype=torch.float).to(device)

    for i in range(graph_size - 1):
        action_prob, test_data = eval(net, config, batch_graphs, selected, test_data, device)

        if i == 0: # In order to calculate the knn matrix only once, the batch size of the first input to the network is 1
            # Expand batch size to sampling size for the next step
            action_prob = action_prob.repeat(1, sampling_size) \
                                     .reshape(-1, graph_size)

            test_data.x = test_data.x.repeat(1, sampling_size, 1) \
                                     .reshape(-1, graph_size, test_data.x.shape[-1])

            test_data.edge_attr = test_data.edge_attr.repeat(1, sampling_size,1,1) \
                                  .reshape(-1, graph_size, test_data.edge_attr.shape[-2], test_data.edge_attr.shape[-1])

            test_data.edge_index = get_edge_index(sampling_size, batch_size, graph_size, \
                                                  config['data_loader']['data']['knn'], \
                                                  test_data.edge_index[1])

            test_data.dis_mat = test_data.dis_mat.repeat(1,sampling_size,1).reshape(-1, graph_size, graph_size)

            batch_graphs = batch_graphs.repeat(1, sampling_size, 1).reshape(-1, graph_size, 2)

        action_prob = torch.exp(action_prob)

        # Probability normalization for all cities
        p_max = torch.max(action_prob, -1)[0]
        action_prob = action_prob / p_max.unsqueeze(-1)

        # A city with a probability of 0 is set to 1e-6
        action_prob = action_prob + 1e-6

        # The selected city is set to 0
        action_prob.scatter_(1, tours.T, 0)

        # Sampling
        selected = action_prob.multinomial(1).squeeze(1)
        tours = torch.cat([tours, selected.unsqueeze(0)])
        dis = test_data.dis_mat.gather(1, tours[-1].unsqueeze(-1).unsqueeze(1) \
                       .expand(test_data.dis_mat.shape[0],1,test_data.dis_mat.shape[1]))
        dis = dis.gather(2, tours[-2].unsqueeze(1).unsqueeze(1)).reshape(-1)
        lens = lens + dis

    # Add the final path
    dis = test_data.dis_mat.gather(1, tours[-1].unsqueeze(-1).unsqueeze(1) \
                   .expand(test_data.dis_mat.shape[0],1,test_data.dis_mat.shape[1]))
    dis = dis.gather(2, tours[0].unsqueeze(1).unsqueeze(1)).reshape(-1)
    lens = lens + dis

    # Choose the shortest path
    lens = lens.reshape(batch_size, sampling_size)
    lens_min = torch.min(lens, dim=1)
 
    tours = tours.T
    tours = tours.reshape(batch_size, sampling_size, -1)
    ids = lens_min[1].unsqueeze(-1).unsqueeze(-1).expand(batch_size,1,graph_size)
    tours_min = tours.gather(1, ids).squeeze(-2) 
 

    return tours_min, lens_min[0]

        
def get_edge_index_beamsearch(beam_size, batch_size, graph_size, k_n, target_ver, prev_k):
    source_ver = torch.ones([batch_size * beam_size * graph_size, k_n], dtype=torch.long) \
                 * torch.arange(0, batch_size * beam_size * graph_size).unsqueeze(1)
    source_ver = source_ver.to(target_ver.device)

    # reduction shift
    shift_dis = torch.ones([batch_size * beam_size, k_n * graph_size], dtype=torch.long) \
                 * torch.arange(0, batch_size * beam_size * graph_size, graph_size).unsqueeze(1)
    shift_dis = shift_dis.reshape(-1).to(target_ver.device)
    target_ver = target_ver - shift_dis
    target_ver = target_ver.reshape(batch_size, beam_size, k_n * graph_size)
    target_ver = target_ver.gather(1, prev_k.unsqueeze(-1) \
                                            .expand(batch_size, beam_size, k_n * graph_size))
    target_ver = target_ver.reshape(-1)

    # Recalculate shift
    shift_dis = torch.ones([batch_size * beam_size, k_n * graph_size], dtype=torch.long) \
                 * torch.arange(0, batch_size * beam_size * graph_size, graph_size).unsqueeze(1)
    shift_dis = shift_dis.reshape(-1).to(target_ver.device)
    target_ver = target_ver + shift_dis

    edge_index = torch.stack([source_ver.reshape(-1), target_ver], dim=0)

    return edge_index


def beamsearch(net, config, batch_graphs, device):
    batch_size, graph_size, _ = batch_graphs.shape
    selected = torch.zeros((batch_size,), dtype=int).to(device)
    test_data = None
    beam_size = config['beamsearch']['size']
    tours = torch.zeros((batch_size, beam_size, 1), dtype=torch.long).to(device)
    lens = torch.zeros((batch_size * beam_size), dtype=torch.float).to(device)
    trans_prob = torch.ones((batch_size, beam_size), dtype=torch.float) 
    trans_prob = trans_prob.to(device)

    for i in range(graph_size - 1):
        action_prob, test_data = eval(net, config, batch_graphs, selected, test_data, device)
        action_prob = torch.exp(action_prob)

        if i == 0:
            beam_action_prob = torch.zeros((batch_size, beam_size * graph_size))
            beam_action_prob[:,:graph_size] = action_prob
            beam_action_prob = beam_action_prob.reshape(-1, graph_size).to(device) # (batch_size * beam_size, graph_size)
            action_prob = beam_action_prob            

            test_data.x = test_data.x.repeat(1, beam_size, 1) \
                                     .reshape(-1, graph_size, test_data.x.shape[-1])

            test_data.edge_attr = test_data.edge_attr.repeat(1, beam_size,1,1) \
                                  .reshape(-1, graph_size, test_data.edge_attr.shape[-2], test_data.edge_attr.shape[-1])

            test_data.edge_index = get_edge_index(beam_size, batch_size, graph_size, \
                                                  config['data_loader']['data']['knn'], \
                                                  test_data.edge_index[1])

            batch_graphs = batch_graphs.repeat(1, beam_size, 1).reshape(-1, graph_size, 2)

        # The selected city is set to 0
        action_prob.scatter_(1, tours.reshape(batch_size * beam_size, -1), 0) # (batch_size * beam_size, graph_size) 
        action_prob = action_prob.reshape(batch_size, beam_size * graph_size)

        # Calculate the cumulative probability
        action_prob *= trans_prob.repeat(1, graph_size)

        # select topk
        best_scores, best_scores_id = action_prob.topk(beam_size, 1, True, True)

        prev_k = best_scores_id / graph_size
        new_nodes = best_scores_id - prev_k * graph_size

        tours = tours.gather(1, prev_k.unsqueeze(-1).expand(batch_size, beam_size, tours.shape[-1]))

        
        test_data.x = test_data.x.reshape(batch_size, beam_size, test_data.x.shape[-2], test_data.x.shape[-1])
        test_data.x = test_data.x.gather(1, prev_k.unsqueeze(-1).unsqueeze(-1) \
                                                  .expand(batch_size, beam_size, test_data.x.shape[-2], test_data.x.shape[-1]))      
        test_data.x = test_data.x.reshape(batch_size * beam_size, test_data.x.shape[-2], test_data.x.shape[-1])

        
        test_data.edge_attr = test_data.edge_attr.reshape(batch_size, beam_size, \
                                                  test_data.edge_attr.shape[-3], test_data.edge_attr.shape[-2], test_data.edge_attr.shape[-1])
        test_data.edge_attr = test_data.edge_attr.gather(1, prev_k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) \
                                                 .expand(batch_size, beam_size, \
                                                  test_data.edge_attr.shape[-3], test_data.edge_attr.shape[-2], test_data.edge_attr.shape[-1]))
        test_data.edge_attr = test_data.edge_attr.reshape(batch_size * beam_size, \
                                                  test_data.edge_attr.shape[-3], test_data.edge_attr.shape[-2], test_data.edge_attr.shape[-1])


        test_data.edge_index = get_edge_index_beamsearch(beam_size, batch_size, graph_size, \
                                                         config['data_loader']['data']['knn'], test_data.edge_index[1], prev_k)

        selected = new_nodes.reshape(-1)
        tours = torch.cat([tours.reshape(batch_size * beam_size, -1), selected.unsqueeze(-1)], -1)
        tours = tours.reshape(batch_size, beam_size, -1)


    # Choose the shortest path
    tours = tours.detach().cpu().numpy()
    test_data.dis_mat = test_data.dis_mat.detach().cpu().numpy()
    costs = []
    for i in range(batch_size):
        c = []
        for j in range(beam_size):
            c.append(test_data.dis_mat[i][tours[i][j], np.roll(tours[i][j], -1)].sum())
        costs.append(c)

    argmin_costs = np.argmin(costs, -1)
    argmin_costs = torch.tensor(argmin_costs).unsqueeze(-1).unsqueeze(-1)

    tours_min = torch.tensor(tours).gather(1, argmin_costs.expand(batch_size, 1, graph_size))
    costs_min = np.min(costs, -1)

    
    return tours_min.squeeze(1), costs_min
    


def test(config):       
    # setup data_loader instances
    graph_size = config['arch']['args']['graph_size']
    test_graphs = load_data('data/test_data/{}/tsp{}/tsp{}_test.txt'\
                              .format(config['data_loader']['data']['graph_type'], \
                                      graph_size, graph_size), graph_size)
    test_gt_paths = np.genfromtxt('data/test_data/{}/tsp{}/tsp{}_test_path.txt'\
                              .format(config['data_loader']['data']['graph_type'], \
                                      graph_size, graph_size), dtype=np.int)

    # load trained model
    model = config.init_obj('arch', module_arch)
    saved_state = torch.load(
            "{}/pretrained_models/model{}_pure_model_best.pth".format(config.save_dir, config.model_id),
            map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state)
    model = model.to(config['gpu_id'])
    print("Load model successfully ~")

    model.eval()

    if config['mode'] == 'greedy':
        result_dir = os.path.join(config.result_dir, "tsp{}/{}/greedy".format(graph_size, config['data_loader']['data']['graph_type']))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
     
        batch_size = config['data_loader']['args']['batch_size']
        batch_num = math.ceil(len(test_graphs) / batch_size)
        tours = []
        with torch.no_grad():
            st = time.time()
            for i in range(batch_num):
                batch_graphs = torch.tensor(test_graphs[i * batch_size: (i+1) * batch_size], dtype=torch.float) \
                                    .to(config['gpu_id'])
                t = greedy_search(model, config, batch_graphs, config['gpu_id'])
                tours.extend(t.detach().cpu().tolist())

            print("Time of divide and conquer: {}".format(time.time() - st))
        save_path(os.path.join(result_dir, 'model{}_greedy_test_path.txt'.format(config.model_id)), test_graphs, np.array(tours), test_gt_paths)
    

    elif config['mode'] == 'sampling':
        result_dir = os.path.join(config.result_dir, "tsp{}/{}/sampling".format(graph_size, config['data_loader']['data']['graph_type']))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        batch_size = config['data_loader']['args']['batch_size']
        batch_num = math.ceil(len(test_graphs) / batch_size)
        tours = []
        lens = []
        with torch.no_grad():
            st = time.time()
            for i in range(batch_num):
                batch_graphs = torch.tensor(test_graphs[i * batch_size: (i+1) * batch_size], dtype=torch.float) \
                                    .to(config['gpu_id'])
                t, l = sampling(model, config, batch_graphs, config['gpu_id'])
                tours.extend(t.detach().cpu().tolist())
                lens.extend(l.detach().cpu().tolist())

            print("Time of divide and conquer: {}".format(time.time() - st))
        save_path(os.path.join(result_dir, 'model{}_sampling{}_test_path.txt'.format(config.model_id, config['sampling']['size'])), test_graphs, np.array(tours), test_gt_paths)


    else:
        result_dir = os.path.join(config.result_dir, "tsp{}/{}/beamsearch".format(graph_size, config['data_loader']['data']['graph_type']))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        batch_size = config['data_loader']['args']['batch_size']
        batch_num = math.ceil(len(test_graphs) / batch_size)
        tours = []
        lens = []
        with torch.no_grad():
            st = time.time()
            for i in range(batch_num):
                batch_graphs = torch.tensor(test_graphs[i * batch_size: (i+1) * batch_size], dtype=torch.float) \
                                    .to(config['gpu_id'])
                t, l = beamsearch(model, config, batch_graphs, config['gpu_id'])
                tours.extend(t.detach().cpu().tolist())
                lens.extend(l)
            print("Time of divide and conquer: {}".format(time.time() - st))
        save_path(os.path.join(result_dir, 'model{}_beamsearch{}_test_path.txt'.format(config.model_id, config['beamsearch']['size'])), test_graphs, np.array(tours), test_gt_paths)




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
    
