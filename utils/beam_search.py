import time
import torch
import numpy as np
import math
from tqdm import tqdm
from env.tsp_env import TSPEnv
from env.graph import Graph
import os
from torch_geometric.data import DataLoader

def eval(env, net, states, device):
    obs = env.get_obs_for_states(states)
    for obs in DataLoader(obs, batch_size=len(obs)):
        obs = obs.to(device)
    priors = net(obs)
    priors = priors.detach().cpu().numpy()

    return priors

def bs(env, net, state, device, k=5):
    all_sequences = []
    sequences = [[state, 0.0, 0]]
    scores = [0 for i in range(state['n_actions'])]
    all_sequences.append(sequences)
    while True:
        all_candidates = []
        for idx, sequence in enumerate(sequences):
            state, score, _ = sequences[idx]
            priors = eval(env, net, [state], device)[0]
            priors = priors[list(state['ava_action'])]
            for logp, action in zip(priors, state['ava_action']):
                all_candidates.append([env.next_state(state, action), score - logp, action])
        sequences = []
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        for i in range(min(k, len(ordered))):
            sequences.append(ordered[i])
            all_sequences.append(sequences)
            scores[ordered[i][2]] = ordered[i][1]

        if env.is_done_state(sequences[0][0]):
            tours = []
            for i in range(len(sequences)):
                tours.append((sequences[i][0]['tour'])) #, env.get_return(sequences[i][0])))
            return tours, env.get_return(sequences[0][0])


class beam_search():
    def __init__(self, config):
        self.config = config
        self.graph_type = config["data_loader"]["data"]["graph_type"]
        self.graph_size = config["arch"]["args"]["graph_size"]
        
    def run(self, model, x, y, device, bs_width=1):
        self.model = model
        self.bs_width = bs_width
        self.device = device

        graph = Graph(len(x), x)
        graph.init()
        env = TSPEnv(graph, self.config)
        state = env.initial_state()
        tours, tour_len = bs(env, self.model, state, self.device, self.bs_width)
        gt_tour_len  = graph.compute_path_len(y.astype(np.int).tolist())

        return tours[0], tour_len, ((tour_len / gt_tour_len) - 1) * 100

