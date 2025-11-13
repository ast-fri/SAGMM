
import torch
import numpy as np
import torch.nn as nn
from dataset import load_dataset
from parse import parser_add_main_args, parse_method 
import argparse
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from data_utils import eval_acc, eval_rocauc, eval_f1, load_fixed_splits
from logger import Logger
from eval import evaluate_large
from graph_positional_encoding import *
import random
import time
import gc


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(123)
def scale_feature(feature, node_feat_mean):
    """Scale structural feature based on node features mean"""
    feature_mean = torch.mean(torch.abs(feature))
    if feature_mean == 0:
        return feature
    scale_factor = node_feat_mean / (feature_mean + 1e-8)
    return feature * scale_factor

def load_pruned_model(args, checkpoint, input_dim, hidden_dim, output_dim, n, device):
    """
    Load a pruned model for "Self_Adaptive_MoE" from a checkpoint.

    This ensures that only active experts are retained based on the checkpoint data.
    """
    full_model = parse_method(args, output_dim, input_dim, n, device)

    if 'remaining_experts' in checkpoint and 'expert_gate_indices' in checkpoint:
        # Access the GMoE component
        gmoe_network = full_model.graph_conv.experts[0]

        # Load checkpoint information
        checkpoint_experts = checkpoint['experts']
        checkpoint_expert_names = checkpoint['experts_names']
        checkpoint_expert_gate_indices = checkpoint['expert_gate_indices']
        remaining_experts = checkpoint['remaining_experts']

        # Debugging information
        # print(f"Loaded checkpoint with {len(checkpoint_experts)} total experts.")
        # print(f"Remaining experts in checkpoint: {remaining_experts}")
        # print(f"Gate indices in checkpoint: {checkpoint_expert_gate_indices}")

        # Identify active experts from the checkpoint
        # print("Check point experts: ", checkpoint_experts)
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

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    lap_save_dir = f"../../lap_matrix//"
    dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    
    dataset.label = dataset.label.to(device)

    # print("Dataset label shape: ", dataset.label.shape)
    
    # get the splits for all runs
    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                        for _ in range(args.runs)]
    elif args.rand_split_class:
        split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                        for _ in range(args.runs)]
    elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
        split_idx_lst = [dataset.load_fixed_splits()
                        for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)

    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    # infer the number of classes for non one-hot and one-hot labels
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    
    # whether or not to symmetrize
    print("Dataset graph: ", dataset.graph["edge_index"].shape)
    if not args.directed and args.dataset != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        needs_directed_model = False
    else:
        needs_directed_model = True

   
    if needs_directed_model:
      
        lap_edge_index = to_undirected(dataset.graph['edge_index'])
        lap_edge_index, _ = remove_self_loops(lap_edge_index)
        # Model: add self-loops to original (will be done after Laplacian computation)
        
    else:
        lap_edge_index, _ = remove_self_loops(dataset.graph['edge_index'])

    if needs_directed_model:
        dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
    else:
        model_edge_index, _ = add_self_loops(lap_edge_index, num_nodes=n)
        dataset.graph['edge_index'] = model_edge_index
    
    print("Dataset graph after undirected: ", dataset.graph["edge_index"].shape)
    
    cpu_device = torch.device("cpu")

    x_agg = get_graph_info(args, lap_save_dir, dataset, d, lap_edge_index, cpu_device)

    if(args.pin_memory):
        x_agg = x_agg.pin_memory()
    ### Loss function (Single-class, Multi-class) ###
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()

    ### Performance metric (Acc, AUC, F1) ###
    if args.metric == 'rocauc':
        eval_func = eval_rocauc
    elif args.metric == 'f1':
        eval_func = eval_f1
    else:
        eval_func = eval_acc
    model_name = "Self_Adaptive_MoE"
    logger = Logger(args.runs, args)
    lap_save_dir = f"../../lap_matrix//"
 
    cpu_device = torch.device("cpu")


    print("Node feat: ", dataset.graph['node_feat'])
    
    checkpoint_dir = f'{args.runs_path}/runs_{args.dataset}/results_{args.folder_name}/checkpoints'
    for run in range(args.runs):
     
        print("*********************")
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(4)
        set_random_seed(args.seed + run)
        checkpoint_path = f"{checkpoint_dir}/checkpoint_{run}.pt"
        is_self_trained = model_name == "Self_Adaptive_MoE"
        
        if is_self_trained:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = load_pruned_model(args, checkpoint, dataset.graph['node_feat'].shape[1],
                                        args.hidden_channels,
                                        max(dataset.label.max().item() + 1, dataset.label.shape[1]),
                                        dataset.graph['num_nodes'], device)
            
        
        model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
        model.eval()
        split_idx = split_idx_lst[0]
      
        
        result = evaluate_large(model, dataset, split_idx, eval_func, criterion, args, x_agg = x_agg)
    
        logger.add_result(run, result[:-1])
        logger.print_statistics(run)
        print_str = f'Train: {100 * result[0]:.2f}%, ' + \
                    f'Valid: {100 * result[1]:.2f}%, ' + \
                    f'Test: {100 * result[2]:.2f}%'
        print(print_str)
        print()
      
    logger.print_statistics()

if __name__ == "__main__":
    main()