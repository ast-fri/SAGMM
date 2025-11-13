import argparse
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph

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
from logger import Logger
from dataset import load_dataset
from data_utils import  eval_acc, eval_rocauc, eval_f1, load_fixed_splits
from eval import evaluate_large
from parse import parse_method, parser_add_main_args
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from ogb.nodeproppred import Evaluator
import numpy as np 
import warnings
import time
import gc
from graph_positional_encoding import * 
from torch_geometric.loader import NeighborLoader
warnings.filterwarnings('ignore')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

large_dataset_training = ["ogbn-papers100M", "ogbn-products"]

def print_memory_summary(device):
    allocated = torch.cuda.memory_allocated(device) / 1e6
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e6
    print(f"Memory Allocated: {allocated:.2f} MB")
    print(f"Max Memory Allocated: {max_allocated:.2f} MB")
    print(torch.cuda.memory_summary(device, abbreviated=True))

def scale_feature(feature, node_feat_mean):
    """Scale structural feature based on node features mean"""
    feature_mean = torch.mean(torch.abs(feature))
    if feature_mean == 0:
        return feature
    scale_factor = node_feat_mean / (feature_mean + 1e-8)
    return feature * scale_factor


def train(model, n, c, x, edge_index, num_batch, train_mask, true_label, criterion, optimizer, device, args, x_agg):
    model.to(device)
    model.train()
    
    idx = torch.randperm(n)
    loss = 0
    for i in range(num_batch):
        idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
        if idx_i.numel() == 0:
            continue  # Skip empty batches
        train_mask_i = train_mask[idx_i]
     
        x_i = x[idx_i].to(device)
        x_agg_i = x_agg[idx_i].to(device)
        edge_index_i, _ = subgraph(idx_i, edge_index, relabel_nodes=True, num_nodes=n)
        edge_index_i = edge_index_i.to(device)
        
        max_index = edge_index_i.max().item()
        min_index = edge_index_i.min().item()
        num_nodes_in_subgraph = x_i.size(0)

        if max_index >= num_nodes_in_subgraph or min_index < 0:
            print(f"Invalid indices in edge_index_i: min {min_index}, max {max_index}")
            print(f"Number of nodes in subgraph: {num_nodes_in_subgraph}")
            continue  # Skip this batch
                
        y_i = true_label[idx_i].to(device)
        optimizer.zero_grad()
        try:
            out_i, aux_loss, _ = model(x_i, edge_index_i, x_agg = x_agg_i)
            
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            print(f"x_i shape: {x_i.shape}, edge_index_i shape: {edge_index_i.shape}")
            exit()  # Skip this batch
        if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
            loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i].to(torch.float))

        else:
            out_i = F.log_softmax(out_i, dim=1)
            loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i])
        loss = loss + aux_loss
        loss.backward()
        optimizer.step()

    return 0, loss

def train_large(model, train_loader, criterion, optimizer, device, args, x_agg):
    model.train()
  
    total_loss = 0
    for i, batch in enumerate(train_loader):
  
        batch = batch.to(device)
        batch.y = batch.y.to(torch.long)
        
        optimizer.zero_grad()
    
        x_agg_batch = x_agg[batch.n_id.cpu()].to(device)
        if hasattr(batch, 'adj_t') and batch.adj_t is not None:
            out, aux_loss, _ = model(batch.x, batch.adj_t, x_agg = x_agg_batch)
        else:
            out, aux_loss, _ = model(batch.x, batch.edge_index, x_agg = x_agg_batch)
        
        
            
        if(args.dataset == "ogbn-papers100M"):
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size].squeeze(1))
        elif(args.dataset == "ogbn-products"):
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, batch.y.squeeze(1))
        
        
        loss = loss + aux_loss
        loss.backward()
        optimizer.step()
        if(args.dataset == "ogbn-papers100M"):
             total_loss += loss.item() * len(batch.input_id)
        else:
             total_loss += loss.item() * batch.x.shape[0]
        del x_agg_batch, out
    return 0, total_loss / len(train_loader.dataset)


@torch.no_grad()
def test_large(model, device, train_loader, val_loader, test_loader, x_agg, args):
    model.eval()
    result = []
    
    def evaluate_loader(loader):
        total_correct = 0
        total_samples = 0
        
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            batch.y = batch.y.to(torch.long)
            x_agg_batch = x_agg[batch.n_id.cpu()].to(device)
            if hasattr(batch, 'adj_t') and batch.adj_t is not None:
                out, _, _ = model(batch.x, batch.adj_t, x_agg = x_agg_batch)
            else:
                out, _, _ = model(batch.x, batch.edge_index, x_agg = x_agg_batch)
            
            if args.dataset == "ogbn-papers100M":
                pred = torch.argmax(out[:batch.batch_size], dim=1, keepdim=True).cpu()
                target = batch.y[:batch.batch_size].cpu()
            else:
                pred = torch.argmax(out, dim=1, keepdim=True).cpu()
                target = batch.y.cpu()
            
            total_correct += (pred == target).sum().item()
            total_samples += target.size(0)
            
            del x_agg_batch, out
            
        
        return total_correct / total_samples if total_samples > 0 else 0.0

    result.append(evaluate_loader(train_loader))
    torch.cuda.empty_cache()
    result.append(evaluate_loader(val_loader))
    torch.cuda.empty_cache()
    result.append(evaluate_loader(test_loader))
    torch.cuda.empty_cache()
    print(result)
    return result


def main(args):
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
    elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        split_idx_lst = [dataset.load_fixed_splits()
                        for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)

    # Basic information of datasets
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    
    print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

    print("Dataset graph: ", dataset.graph["edge_index"].shape)
    if not args.directed and args.dataset != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        needs_directed_model = False
    else:
        needs_directed_model = True

    # Prepare Laplacian edge_index (clean, no self-loops)
    if needs_directed_model:
        # Case: Model is directed, Laplacian needs undirected
        # Create undirected version only for Laplacian
        lap_edge_index = to_undirected(dataset.graph['edge_index'])
        lap_edge_index, _ = remove_self_loops(lap_edge_index) 
    else:
        # Case: Both model and Laplacian use undirected
        # Remove self-loops for Laplacian, reuse result for model
        lap_edge_index, _ = remove_self_loops(dataset.graph['edge_index'])


    if needs_directed_model:
        # Model uses original directed structure + self-loops
        dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
    else:
        model_edge_index, _ = add_self_loops(lap_edge_index, num_nodes=n)
        dataset.graph['edge_index'] = model_edge_index
    
    print("Dataset graph after undirected: ", dataset.graph["edge_index"].shape)
    
    
    # Step 3: Ensure CPU placement for Laplacian computation (single device transfer)
    cpu_device = torch.device("cpu")

    # Step 4: Compute Laplacian PE on CPU
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

    logger = Logger(args.runs, args)
  
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
    else:
        true_label = dataset.label

    
    with open(result_save_dir+'/results_best_val.txt', 'w') as f:
        f.write("Run\tEpoch\tLoss\tTrain\tValid\tTest\n")
    with open(result_save_dir+'/results_all.txt', 'w') as f:
        f.write("Run\tEpoch\tLoss\tTrain\tValid\tTest\tTime_Taken_So_Far (in Minutes)\n")
    
    
    if(args.dataset in large_dataset_training):
        dataset._data.edge_index = dataset.graph['edge_index']
        dataset.convert_to_adj_t_after_preprocessing()

    # exit()
    
    start_time=time.time()
    val_acc_per_expert = {}  # Stores best validation accuracy per expert count
    test_acc_per_expert = {}  # Stores corresponding test accuracy
    train_acc_per_expert = {}
    if(args.dataset in large_dataset_training):
        evaluator = Evaluator(name=args.dataset)
    ### Training loop ###
    for run in range(args.runs):
        print(f"************ Run{run} *************")
        set_random_seed(args.seed + run)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(4)
   
        train_losses = []
        valid_accuracies = []
        test_accuracies = []
        epoch_list=[]
        model = parse_method(args, c, d, n, device)
     
        if(run == 0):
            print('MODEL:', model)
        print()
        
        
        if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[run]

        if(args.dataset in large_dataset_training):
            fan_out_list = list(map(int, args.fan_out.split(',')))

            kwargs = dict(
                num_neighbors=fan_out_list,
                batch_size=args.batch_size,
            )
            print("Split idx train shape: ", split_idx['train'].shape)
            train_loader = NeighborLoader(dataset.pyg_data, input_nodes=split_idx['train'], shuffle=True, drop_last=True, 
                                          num_workers=12, **kwargs)
            val_loader = NeighborLoader(dataset.pyg_data, input_nodes=split_idx['valid'], num_workers=12,
                                        **kwargs)
            test_loader = NeighborLoader(dataset.pyg_data, input_nodes=split_idx['test'], num_workers=12,
                                        **kwargs)

              
        
        else:
            train_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[split_idx['train']] = True
          
            num_batch = n // args.batch_size + (n % args.batch_size > 0)

        gmoe_network = model.network_model.experts[0]
        optimizer = torch.optim.Adam(
            model.parameters(), weight_decay=args.gnn_weight_decay, lr=args.lr)
        best_val = 0
        best_test = 0
        best_train = 0
        current_val_accuracy = 0
        total_epoch_time = 0
        total_prune_time = 0
        total_eval_time = 0
        for epoch in range(args.epochs):
            
            epoch_start = timer()
            try:
                if(args.dataset in large_dataset_training):
                    out, loss = train_large(model, train_loader, criterion, optimizer, device, args, x_agg)
                else:
                    out, loss = train(model, n, c, dataset.graph["node_feat"], dataset.graph["edge_index"], num_batch, train_mask, true_label, criterion, optimizer, device, args, x_agg)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("Caught OOM error!")
                    print_memory_summary(device)
                else:
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
            if epoch % args.eval_step == 0:
          
                if(args.dataset in large_dataset_training):
                    result = test_large(model, device, train_loader, val_loader, test_loader, x_agg, args)
                else:
                    result = evaluate_large(model, dataset, split_idx, eval_func, args, x_agg=x_agg)
                
                if(args.dataset in large_dataset_training):
                    train_losses.append(loss)
                else:
                    train_losses.append(loss.item())
                valid_accuracies.append(result[1])
                current_val_accuracy = result[1]
                test_accuracies.append(result[2])
                epoch_list.append(epoch)
                end_time=time.time()
                with open(result_save_dir+'/results_all.txt', 'a') as f:
                    f.write(f"{run}\t{epoch}\t{loss:.4f}\t{result[0]:.4f}\t{result[1]:.4f}\t{result[2]:.4f}\t\t{(end_time-start_time)/60:.2f}\n")
                # Save checkpoint if validation accuracy improved
                if best_val == 0:
                    best_val = result[1]
                if result[1] >= best_val:
                    best_val = result[1]
                    best_test = result[2]
                    best_train = result[0]
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{run}.pt')
                    expert_state_dicts = {}
                    
                    for i in range(args.max_experts):
                        if(gmoe_network.experts[i] is not None):
                            expert_state_dicts[i] = gmoe_network.experts[i].state_dict()

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'best_val': best_val,
                        'remaining_experts': gmoe_network.num_experts,
                        'expert_gate_indices': gmoe_network.expert_gate_indices,  # Keep track of active experts
                        'experts': gmoe_network.experts,
                        'experts_names': gmoe_network.expert_names,
                        'expert_state_dict': expert_state_dicts,   
                    }, checkpoint_path)
                    # Save results in text file
                    with open(result_save_dir+'/results_best_val.txt', 'a') as f:
                        f.write(f"{run}\t{epoch}\t{loss:.4f}\t{result[0]:.4f}\t{result[1]:.4f}\t{result[2]:.4f}\n")
                
                if epoch % args.display_step == 0:
                    print_str = f'Epoch: {epoch:02d}, ' + \
                                f'Loss: {loss:.4f}, ' + \
                                f'Train: {100 * result[0]:.2f}%, ' + \
                                f'Valid: {100 * result[1]:.2f}%, ' + \
                                f'Test: {100 * result[2]:.2f}%'
                    print(print_str)
                    print()
            eval_end_time = timer()

            prune_start_time = timer()
            if (not args.stop_prune and ((epoch + 1) % args.prune_interval == 0)):
              

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

                print(f"Importance values: {valid_importance.tolist()}")
                print(f"Threshold: {threshold:.4f}")
                print(f"Num experts to remove: {num_experts_to_remove}")

                # Prune experts
                if num_experts_to_remove > 0:
                    val_acc_per_expert[gmoe_network.num_experts] = best_val*100
                    test_acc_per_expert[gmoe_network.num_experts] = best_test*100
                    train_acc_per_expert[gmoe_network.num_experts] = best_train*100
                    best_train = 0
                    best_val = 0
                    best_test = 0
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
            del out, loss
            epoch_end = timer()
            total_epoch_time = total_epoch_time + (epoch_end - epoch_start)
            total_prune_time = total_prune_time + (prunt_end_time - prune_start_time)
            total_eval_time = total_eval_time + (eval_end_time - eval_start_time)
            # print(f"Epoch time: {epoch_end - epoch_start}, Prune time: {prunt_end_time-prune_start_time}, Eval time: {eval_end_time-eval_start_time}")
            
        # print(f"Total epoch time: {total_epoch_time}, Total prune time: {total_prune_time}, Total eval time: {total_eval_time}")
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
        # Clear model and optimizer
        del model, optimizer

        
        torch.cuda.empty_cache()
        gc.collect()
        

    logger.print_statistics()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
