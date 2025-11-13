import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import torch.nn as nn
import torch.nn.functional as F
from dataset import load_dataset  # To load the dataset
from parse import parser_add_main_args, parse_method  # Argument parsing
import argparse
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, LineString
from scipy.spatial import Delaunay
import alphashape 
import matplotlib.pyplot as plt
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from logger import Logger
from eval import evaluate_link
from timeit import default_timer as timer
from graph_positional_encoding import *
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import pstats
import io
from pynvml import *
import threading
prof = 0
import psutil, time
import gc
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# set_random_seed(123)

def util(stage, args):
    nvmlInit()
    gpu = args.device

    f = open(f'gpu_usage_{args.dataset}_infer_usage.txt', "a+")
    handle = nvmlDeviceGetHandleByIndex(gpu)
    f.write(str(stage) + ",")
    global prof

    while prof == 0:
        util = nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        f.write(str(mem_info.used >> 20) + ',')
        time.sleep(0.1)
    
    f.write("\n\nEND\n\n")
    nvmlShutdown()
    f.close()

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
  

def load_pruned_model(args, checkpoint, input_dim, hidden_dim, output_dim, n, device):
    """
    Load a pruned model for "Self_Adaptive_MoE" from a checkpoint.

    This ensures that only active experts are retained based on the checkpoint data.
    """
    full_model = parse_method(args, 0, input_dim, n, device)

    if 'remaining_experts' in checkpoint and 'expert_gate_indices' in checkpoint:
        # Access the GMoE component
        gmoe_network = full_model.graph_conv.experts[0]

        # Load checkpoint information
        checkpoint_experts = checkpoint['experts']
        checkpoint_expert_names = checkpoint['experts_names']
        checkpoint_expert_gate_indices = checkpoint['expert_gate_indices']
        remaining_experts = checkpoint['remaining_experts']

        active_expert_indices = [i for i, expert in enumerate(checkpoint_experts) if expert is not None]
        # print(f"Active expert indices from checkpoint: {active_expert_indices}")

        # Remove experts that are not active
        for idx in range(len(gmoe_network.experts)):
            if idx not in active_expert_indices:
                # print(f"Pruning expert at index {idx}.")
                gmoe_network.remove_expert(expert_index=idx, optimizer=None)

        # Ensure the gate indices align with the checkpoint
        gmoe_network.expert_gate_indices = checkpoint_expert_gate_indices

        # Debugging after pruning
        active_experts_post_pruning = [
            i for i, expert in enumerate(gmoe_network.experts) if expert is not None
        ]
        print(f"Active experts after pruning: {active_experts_post_pruning}")
        # print(f"Number of active experts: {len(active_experts_post_pruning)}")

    return full_model


def main():
    # np.random.seed(42)
    parser = argparse.ArgumentParser(description='Prediction Analysis Script')
    parser_add_main_args(parser)
    parser.add_argument('--reduction_method', type=str, default='umap',
                        choices=['pca', 'tsne', 'umap'],
                        help='Dimensionality reduction method')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of nodes to sample for plotting')
    args = parser.parse_args()
    lap_save_dir = f"../../lap_matrix/"
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)
    data = dataset[0]
    if(args.dataset == 'ogbl-ddi'):
        emb = torch.nn.Embedding(data.adj_t.size(0),
                                args.hidden_channels).to(device)
        data.x = emb.weight
    elif(args.dataset == 'ogbl-collab'):
        import torch_geometric.transforms as T
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        data = T.ToSparseTensor()(data)
    elif(args.dataset == "ogbl-citation2"):
        data.adj_t = data.adj_t.to_symmetric()
    else:
        data.x = data.x.to(torch.float)

    adj_t = data.adj_t
    # data = data.to(device)
    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    # pre compute GCN normalization
    if(args.dataset == "ogbl-ppa"):
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t_norm = adj_t
        
    data = data.to(device)

    undir_edge_index = to_undirected(edge_index)
    lap_edge_index, _ = remove_self_loops(undir_edge_index)
    
   
    
    split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]
    
    ### Basic information of datasets ###
    n = dataset.num_nodes
    e = adj_t.size(0)
    if(args.dataset == 'ogbl-ddi'):
        d = args.hidden_channels
    else:
        d = data.num_features
    print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num features {d} ")

    # Compute PE
    x_agg = get_graph_info(args, lap_save_dir, dataset, d, lap_edge_index, data.x, lap_edge_index, device)
    
    # print("Laplacian created")
    # exit()
        
    # For ogbl citation it is not hits@K metric but to keep code structure is kept same so K=100
    if(args.dataset == 'ogbl-ddi'):
        K = 20
    elif(args.dataset == 'ogbl-collab'):
        K = 50   
    else:
        K = 100

    model_name = "Self_Adaptive_MoE"
    loggers = {
        f'Hits@{K}': Logger(args.runs, args)
    }
   
    checkpoint_dir = f'{args.runs_path}/runs_{args.dataset}/results_{args.folder_name}/checkpoints'
    
    evaluator = Evaluator(name=args.dataset)
    for run in range(args.runs):
        
        print("*********************")
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(4)
        set_random_seed(args.seed + run)
        th1 = threading.Thread(target=util, args=(str(run) + "---", args))
        global prof
        prof = 0
        th1.start()
        predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                                    args.gnn_num_layers, args.gnn_dropout).to(device)
        
        checkpoint_path = f"{checkpoint_dir}/checkpoint_{run}.pt"
        is_self_trained = model_name == "Self_Adaptive_MoE"
        
        if is_self_trained:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = load_pruned_model(args, checkpoint, d,
                                        args.hidden_channels,
                                        args.hidden_channels,
                                        n, device)
            
        model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'], strict=False)
        model.eval()
        # print(model)
        # print(torch.load(checkpoint_path)['embedding'])
        
        predictor.load_state_dict(torch.load(checkpoint_path)['predictor'])
        predictor.eval()
        split_idx = split_idx_lst[run]
        eval_start = timer()
        result =  evaluate_link(model, predictor, data, split_idx, evaluator, args.batch_size, args.method, device, ep=None, x_agg=x_agg, K=K, dataset=args.dataset)
        eval_end = timer()
        # print("result: ", result)
        print("Infer time: ", eval_end-eval_start)
        prof = 1
        th1.join()
        # logger.add_result(run, result[:-1])
        # logger.print_statistics(run)
        # print_str = f'Train: {100 * result[0]:.2f}%, ' + \
        #             f'Valid: {100 * result[1]:.2f}%, ' + \
        #             f'Test: {100 * result[2]:.2f}%'
        # print(print_str)
        print(result)
        print()
    # logger.print_statistics()

if __name__ == "__main__":
    main()