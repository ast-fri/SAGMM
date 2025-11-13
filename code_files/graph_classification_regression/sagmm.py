import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from network_model import NetworkModel
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from model_conv import GNN_node_Virtualnode
class SAGMM(nn.Module):
    def __init__(self, args, num_tasks, hidden_channels, out_channels,
                 gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True, gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.8, aggregate='add',num_experts=8, graph_pooling = "mean", deg=None):
        super().__init__()
        self.args = args
        if args.expert_type == "multiple_models":
            print("Multiple models (GCN,GIN, SAGE, GAT) as experts selected")
            self.network_model=NetworkModel(args, num_tasks, hidden_channels, out_channels, gnn_num_layers, gnn_dropout, num_experts, deg)
        elif args.expert_type == "sage":
            print("GraphSAGE model selected as expert")
            self.network_model = GNN_node_Virtualnode(gnn_num_layers, hidden_channels, drop_ratio = gnn_dropout, 
                     gnn_type = "sage")
        elif args.expert_type == "gcn":
            print("GCN model selected as expert")
            self.network_model = GNN_node_Virtualnode(gnn_num_layers, hidden_channels, drop_ratio = gnn_dropout, 
                     gnn_type = "gcn")
        elif args.expert_type == "gat":
            print("GAT model selected as expert")
            self.network_model = GNN_node_Virtualnode(gnn_num_layers, hidden_channels, drop_ratio = gnn_dropout, 
                     gnn_type = "gat")
        elif args.expert_type == "gin":
            print("GIN model selected as expert")
            self.network_model = GNN_node_Virtualnode(gnn_num_layers, hidden_channels, drop_ratio = gnn_dropout, 
                     gnn_type = "gin")
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.aggregate = aggregate
        self.args = args
        self.graph_pooling = graph_pooling
        self.num_tasks = num_tasks

        if aggregate == 'add':
            self.fc = nn.Linear(1* hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(1 * 1 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(hidden_channels, 2*hidden_channels), torch.nn.BatchNorm1d(2*hidden_channels), torch.nn.ReLU(), torch.nn.Linear(2*hidden_channels, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(hidden_channels, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*hidden_channels, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(hidden_channels, self.num_tasks)
        self.params = list(self.network_model.parameters()) if self.network_model is not None else []
        self.params.extend(list(self.fc.parameters()))

    def forward(self, batched_data, x_agg=None):
        if self.args.expert_type == "multiple_models":
            x, loss, raw_logits = self.network_model(batched_data, x_agg) #
            x=self.fc(x)
            pred = self.graph_pred_linear(x)
            return pred, loss, raw_logits
        else:
            x = self.network_model(batched_data) #
            x = self.pool(x, batched_data.batch)
            pred = self.graph_pred_linear(x)
            return pred
    def reset_parameters(self):
        # Reset graph convolutional layer parameters
        self.network_model.reset_parameters()
        # Reset fully connected layers
        self.fc.reset_parameters()
        self.graph_pred_linear.reset_parameters()    
        # Reset pooling layer 
        if hasattr(self.pool, 'reset_parameters'):
            self.pool.reset_parameters()
        # Reset parameters for any additional submodules
        if isinstance(self.pool, GlobalAttention):
            for layer in self.pool.gate_nn:
                if isinstance(layer, torch.nn.BatchNorm1d):
                    layer.reset_parameters()
