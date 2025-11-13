import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
import time
import matplotlib.pyplot as plt
from logger import Logger
import warnings

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, eval_f1, load_fixed_splits
from eval import evaluate
from parse import parse_method, parser_add_main_args
from timeit import default_timer as timer
import gc
from graph_positional_encoding import * 
warnings.filterwarnings('ignore')

def print_memory_summary(device):
    allocated = torch.cuda.memory_allocated(device) / 1e6
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e6
    print(f"Memory Allocated: {allocated:.2f} MB")
    print(f"Max Memory Allocated: {max_allocated:.2f} MB")
    print(torch.cuda.memory_summary(device, abbreviated=True))

def train(model, dataset, train_idx, criterion, optimizer, args, x_agg):
    model.train()
    optimizer.zero_grad()
    out, aux_loss, log_info = model(dataset.graph['node_feat'], dataset.graph['edge_index'], x_agg=x_agg)
    
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        loss = criterion(out[train_idx], true_label.squeeze(1)[
            train_idx].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
    loss = loss + aux_loss
    loss.backward()
    optimizer.step()
    return out, loss, log_info


def main(args):

    # Set initial seed for data loading and preprocessing
    set_random_seed(args.seed)
    if(args.epochs % args.prune_interval == 0):
        print("!!!! Warning, do not keep prune interval as multiple of epoch !!!! ")
    checkpoint_dir = f'{args.runs_path}/runs_{args.dataset}/results_{args.folder_name}/checkpoints'
    result_save_dir=f"{args.runs_path}/runs_{args.dataset}/results_{args.folder_name}"
    log_file_dir = f"{args.runs_path}/parameters/{args.dataset}/"
    lap_save_dir = f"../../lap_matrix//"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_save_dir, exist_ok=True)
    os.makedirs(log_file_dir, exist_ok=True)
    os.makedirs(lap_save_dir, exist_ok=True)

    if(args.gate_method):
        log_file = log_file_dir + f"{args.dataset}_{args.method}_{args.gate_method}_{args.folder_name}.txt"
    else:
        log_file = log_file_dir + f"{args.dataset}_{args.method}.txt"
    with open(log_file, "w") as file:
        file.write("epoch\ttotal_params\ttrainable_params\n")

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        print("device: ",device)
        
    ### Load and preprocess data ###
    dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    # get the splits for all runs
    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                        for _ in range(args.runs)]
    elif args.rand_split_class:
        split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                        for _ in range(args.runs)]
    elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
        print("OGB splits")
        split_idx_lst = [dataset.load_fixed_splits()
                        for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)

    ### Basic information of datasets ###
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    # infer the number of classes for non one-hot and one-hot labels
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

    
    
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
    # Laplacian needs undirected, model keeps original structure
    undir_edge_index = to_undirected(dataset.graph['edge_index'])
    lap_edge_index, _ = remove_self_loops(undir_edge_index)
    model_edge_index, _ = add_self_loops(lap_edge_index, num_nodes=n)
   
    # Update dataset
    dataset.graph['edge_index'] = model_edge_index.to(device)
    
    # Compute PE
    x_agg = get_graph_info(args, lap_save_dir, dataset, d, lap_edge_index, device)
   
   
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

    logger = Logger(args.runs, args)
   
    
    with open(result_save_dir+'/results_best_val.txt', 'w') as f:
        f.write("Run\tEpoch\tLoss\tTrain\tValid\tTest\n")
    with open(result_save_dir+'/results_all.txt', 'w') as f:
        f.write("Run\tEpoch\tLoss\tTrain\tValid\tTest\tTime_Taken_So_Far (in Minutes)\n")

    start_time=time.time()
   
    val_acc_per_expert = {}  # Stores best validation accuracy per expert count
    test_acc_per_expert = {}  # Stores corresponding test accuracy
    train_acc_per_expert = {}
    
    ### Training loop ###
    for run in range(args.runs):
        log_per_epoch = []
        set_random_seed(args.seed+run)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
       
        train_losses = []
        valid_accuracies = []
        test_accuracies = []
        epoch_list=[]
        
        model = parse_method(args, c, d, n, device)
        if(run == 0):
            print('MODEL:', model)
        print()
        print(f"************ Run{run} *************")
        
        if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        
        
        optimizer = torch.optim.Adam(
            model.parameters(), weight_decay=args.gnn_weight_decay, lr=args.lr)
        
        # Initialize 
        gmoe_network = model.network_model.experts[0]
    
        best_val = 0
        best_test = 0
        best_train = 0
        current_val_accuracy = 0
        total_epoch_time = 0
        total_prune_time = 0
        total_eval_time = 0
        
        for epoch in range(args.epochs):
            # print(f'***** Epoch: {epoch:02d} ******')
            epoch_start = timer()
            try:
                out, loss, log_info = train(model, dataset, train_idx, criterion, optimizer, args, x_agg)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("Caught OOM error!")
                    print_memory_summary(device)
                print(e)
                exit() 
            
           # Track model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
             
            # Log to file
            with open(log_file, "a") as file: 
                file.write(f"{epoch}\t{total_params}\t{trainable_params}\n")   
            
            
            # Evaluation
            eval_start_time = timer()  
            if (epoch) % args.eval_step == 0:
           
                result = evaluate(model, dataset, split_idx, eval_func, args, x_agg=x_agg)
            
                train_losses.append(loss.item())
                valid_accuracies.append(result[1])
                current_val_accuracy = result[1]
                test_accuracies.append(result[2])
                epoch_list.append(epoch)
                end_time=time.time()
                with open(result_save_dir+'/results_all.txt', 'a') as f:
                    f.write(f"{run}\t{epoch}\t{loss.item():.4f}\t{result[0]:.4f}\t{result[1]:.4f}\t{result[2]:.4f}\t\t{(end_time-start_time)/60:.2f}\n")
                # Save checkpoint if validation accuracy improved
                if best_val == 0:
                    best_val = result[1]
                if result[1] >= best_val:
                    best_val = result[1]
                    best_test = result[2]
                    best_train = result[0]
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{run}.pt')
                    expert_state_dicts = {}
                    # print("test accuracy: ", result[2])
                    for i in range(args.max_experts):
                        if(gmoe_network.experts[i] is not None):
                            expert_state_dicts[i] = gmoe_network.experts[i].state_dict()
                        
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'best_val': best_val,
                        'embedding': out,
                        'remaining_experts': gmoe_network.num_experts,
                        'expert_gate_indices': gmoe_network.expert_gate_indices,  # Keep track of active experts
                        'experts': gmoe_network.experts,
                        'experts_names': gmoe_network.expert_names,
                        'expert_state_dict': expert_state_dicts,
                        }, checkpoint_path)
                    # Save results in text file
                    with open(result_save_dir+'/results_best_val.txt', 'a') as f:
                        f.write(f"{run}\t{epoch}\t{loss.item():.4f}\t{result[0]:.4f}\t{result[1]:.4f}\t{result[2]:.4f}\n")
                    
                if epoch % args.display_step == 0:
                    print_str = f'Epoch: {epoch:02d}, ' + \
                                f'Loss: {loss.item():.4f}, ' + \
                                f'Train: {100 * result[0]:.2f}%, ' + \
                                f'Valid: {100 * result[1]:.2f}%, ' + \
                                f'Test: {100 * result[2]:.2f}%'
                    print(print_str)
                    # print()
            eval_end_time = timer()

            # Pruning mechanism
            prune_start_time = timer()
            if (not args.stop_prune and ((epoch + 1) % args.prune_interval == 0)):
                print(f"Inside expert removal process")

                # Compute the importance metrics and filter valid contributions
                valid_mask = (gmoe_network.ema_contributions != float('-inf'))
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                valid_importance = (gmoe_network.ema_contributions / n)[valid_indices]

                print(f"Current val accuracy: {current_val_accuracy:.4f}, Best Val Accuracy: {best_val:.4f}")

                # Dynamically adjust the threshold factor
                if current_val_accuracy < best_val:
                    print("Validation accuracy dropped. Reducing pruning rate.")
                    threshold_factor = max(args.importance_threshold_factor * 0.75, 0.01)
                elif current_val_accuracy > best_val:
                    print("Validation accuracy improved. Increasing pruning rate.")
                    threshold_factor = min(args.importance_threshold_factor * 1.5, 2.0)
                    # best_val = current_val_accuracy
                else:
                    threshold_factor = min(args.importance_threshold_factor*1.2, 0.8)

                
                # Compute the pruning threshold
                if valid_importance.numel() > 0:
                    mean_importance = valid_importance.mean()
                    threshold = mean_importance * threshold_factor
                else:
                    threshold = float('inf')  # Prevent any experts from being removed

                # Identify experts to remove
                experts_to_remove_mask = valid_importance < threshold
                experts_to_remove_indices = valid_indices[experts_to_remove_mask]

                # Ensure at least min_experts remain
                num_experts_to_remove = min(
                    experts_to_remove_indices.numel(),
                    gmoe_network.num_experts - args.min_experts
                )
                experts_to_remove_indices = experts_to_remove_indices[:num_experts_to_remove]

               
                # Prune experts
                if num_experts_to_remove > 0:
                    val_acc_per_expert[gmoe_network.num_experts] = best_val*100
                    test_acc_per_expert[gmoe_network.num_experts] = best_test*100
                    train_acc_per_expert[gmoe_network.num_experts] = best_train*100
                    best_val = 0
                    best_test = 0
                    best_train = 0
                    print("Pruning experts...")
                    for expert_index in experts_to_remove_indices.tolist():
                        expert_list_index = gmoe_network.expert_gate_indices.index(expert_index)
                        print(f"Removed expert '{gmoe_network.expert_names[expert_list_index]}' "
                            f"(gate index {expert_index}). Total experts: {gmoe_network.num_experts - 1}")
                        optimizer = gmoe_network.remove_expert(expert_index=expert_list_index, optimizer=optimizer)

                    # Log remaining experts
                    print(f"Remaining experts: {gmoe_network.expert_names}")
                else:
                    print("No experts removed in this epoch.")
                
            prunt_end_time = timer()
            epoch_end = timer()
            
           
            
            total_epoch_time = total_epoch_time + (epoch_end - epoch_start)
            total_prune_time = total_prune_time + (prunt_end_time - prune_start_time)
            total_eval_time = total_eval_time + (eval_end_time - eval_start_time)
            # print(f"Epoch time: {epoch_end - epoch_start}, Prune time: {prunt_end_time-prune_start_time}, Eval time: {eval_end_time-eval_start_time}")
        # Save plot
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(epoch_list,train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(122)
        plt.plot(epoch_list,valid_accuracies, label='Validation Accuracy')
        plt.plot(epoch_list,test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig(f'{result_save_dir}/plot_run_{run}.png')
        plt.close()
        
        print("Experts left: ", gmoe_network.expert_names)
        final_result = [best_train, best_val, best_test, 0, 0]
        logger.add_result(run, final_result[:-1])
        logger.print_statistics(run)
        
    logger.print_statistics()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)

