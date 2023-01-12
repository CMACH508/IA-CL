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
from utils.utils import save_path, load_data
import torch.multiprocessing as mp
from scipy.spatial import distance_matrix



def _calc_insert_cost(D, prv, nxt, ins):
    """
    Calculates insertion costs of inserting ins between prv and nxt
    :param D: distance matrix
    :param prv: node before inserted node, can be vector
    :param nxt: node after inserted node, can be vector
    :param ins: node to insert
    :return:
    """
    return (
        D[prv, ins]
        + D[ins, nxt]
        - D[prv, nxt]
    )


def run_insertion(loc, gt_path, method):
    n = len(loc)
    D = distance_matrix(loc, loc)

    mask = np.zeros(n, dtype=bool)
    tour = []  # np.empty((0, ), dtype=int)
    for i in range(n):
        feas = mask == 0
        feas_ind = np.flatnonzero(mask == 0)
        if method == 'random':
            # Order of instance is random so do in order for deterministic results
            a = i
        elif method == 'nearest':
            if i == 0:
                a = 0  # order does not matter so first is random
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmin()] # node nearest to any in tour
        elif method == 'cheapest':
            assert False, "Not yet implemented" # try all and find cheapest insertion cost

        elif method == 'farthest':
            if i == 0:
                a = D.max(1).argmax()  # Node with farthest distance to any other node
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmax()]  # node which has closest node in tour farthest
        mask[a] = True

        if len(tour) == 0:
            tour = [a]
        else:
            # Find index with least insert cost
            ind_insert = np.argmin(
                _calc_insert_cost(
                    D,
                    tour,
                    np.roll(tour, -1),
                    a
                )
            )
            tour.insert(ind_insert + 1, a)

    cost = D[tour, np.roll(tour, -1)].sum()
    gt_cost = D[gt_path, np.roll(gt_path, -1)].sum()
    gap = cost / gt_cost - 1

    return cost, gap if gap > 0 else 0

def nearest_neighbour(graph_size, loc, gt_path, res):
    batch_size = 1
    D = distance_matrix(loc, loc)
    gt_cost = D[gt_path, np.roll(gt_path, -1)].sum()

    D = torch.tensor([D])
    
    current = torch.tensor([0])
    dist_to_startnode = torch.gather(D, 2, current.view(-1, 1, 1).expand(batch_size, graph_size, 1)).squeeze(2)
    tour = [current]

    total_dist = 0
    for i in range(graph_size - 1):
        # Mark out current node as option
        D.scatter_(2, current.expand(batch_size, graph_size, 1), np.inf)
        nn_dist = torch.gather(D, 1, current.view(-1, 1, 1).expand(batch_size, 1, graph_size)).squeeze(1)

        min_nn_dist, current = nn_dist.min(1)
        total_dist += min_nn_dist
        tour.append(current)

    total_dist += torch.gather(dist_to_startnode, 1, current.view(-1, 1)).squeeze(1)   
    gap = total_dist.item() / gt_cost - 1         

    return total_dist.item(), gap if gap > 0 else 0


def solve_insertion(graph_size, graphs, gt_paths, method, res):
    start = time.time()
    total_cost = 0
    total_gap = 0
    for graph, gt_path in zip(graphs, gt_paths):
        if method == 'nn':
            cost, gap = nearest_neighbour(graph_size, graph, gt_path, res)
        else:
            cost, gap = run_insertion(graph, gt_path, method)

        total_cost += cost
        total_gap += gap
    duration = time.time() - start
    res.put((total_cost, total_gap, duration))        


def test(opts):       
    # setup data_loader instances
    test_graphs = load_data('data/test_data/random/tsp{}/tsp{}_test.txt'\
                              .format(opts.graph_size, opts.graph_size), opts.graph_size)
    test_gt_paths = np.genfromtxt('data/test_data/random/tsp{}/tsp{}_test_path.txt'\
                              .format(opts.graph_size, opts.graph_size)).astype(int)

    res = mp.Queue()

    processes = []
    
    tasks_num = len(test_graphs) // opts.n_worker
    extra_num = len(test_graphs) % opts.n_worker
 

    for idx in range(opts.n_worker):
        if idx == opts.n_worker - 1:
            graphs = test_graphs[idx * tasks_num: (idx + 1) * tasks_num + extra_num]
            gt_paths = test_gt_paths[idx * tasks_num: (idx + 1) * tasks_num + extra_num]
        else:
            graphs = test_graphs[idx * tasks_num: (idx + 1) * tasks_num]
            gt_paths = test_gt_paths[idx * tasks_num: (idx + 1) * tasks_num]

        p = mp.Process(target=solve_insertion, args=(opts.graph_size, graphs, gt_paths, opts.method, res))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()
    
    total_cost = 0
    total_gap = 0
    max_duration = float('inf')
    for i in range(len(processes)):
        cost, gap, duration = res.get()
        total_cost += cost
        total_gap += gap
        max_duration = max(max_duration, duration)
    print([total_cost / 10000, total_gap / 100, duration])


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('method', default='random', help="Name of the method to evaluate, 'nn', 'gurobi' or 'nearest|random|farthest'")
    args.add_argument('graph_size', default=20, type=int, help="number of cities")
    args.add_argument('n_worker', default=16, type=int, help="number of workers")
    opts = args.parse_args()

    test(opts)

