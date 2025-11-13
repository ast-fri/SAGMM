import argparse, os, math
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger
from sagmm_gating import SAGMM_network
import numpy as np 

class NetworkModel(torch.nn.Module):
    def __init__(self, args, num_tasks, hidden_channels, out_channels, num_layers,
                 dropout, num_experts=4, deg=None):
        super(NetworkModel, self).__init__()
        self.num_layers = num_layers
        self.experts = torch.nn.ModuleList()
        ######### code with multiple models as experts
        networks = SAGMM_network(args,num_tasks, hidden_channels, out_channels, num_layers, dropout, num_experts, deg)
        self.experts.append(networks)

    def reset_parameters(self):
        # print("Multiple models reset")
        for conv in self.experts:
            conv.reset_parameters()

    def forward(self, batched_data, x_agg = None):
        self.load_balance_loss = 0 # initialize load_balance_loss to 0 at the beginning of each forward pass.
        for expert in self.experts:
            if isinstance(expert, SAGMM_network):
                x, _layer_load_balance_loss, raw_logits = expert(batched_data, x_agg)
                self.load_balance_loss += _layer_load_balance_loss
        self.load_balance_loss = 0.0
        return x, self.load_balance_loss, raw_logits
