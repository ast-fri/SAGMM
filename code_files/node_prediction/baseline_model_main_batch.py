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
import time
from logger import Logger
from dataset import load_dataset
from data_utils import  eval_acc, eval_rocauc, eval_f1, load_fixed_splits
from eval import evaluate_large
from parse import parse_method, parser_add_main_args
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from ogb.nodeproppred import Evaluator
import warnings
import gc
from graph_positional_encoding import * 
from torch_geometric.loader import NeighborLoader
warnings.filterwarnings('ignore')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

large_dataset_training = ["ogbn-papers100M","ogbn-products"] # 

def print_memory_summary(device):
    allocated = torch.cuda.memory_allocated(device) / 1e6
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e6
    print(f"Memory Allocated: {allocated:.2f} MB")
    print(f"Max Memory Allocated: {max_allocated:.2f} MB")
    print(torch.cuda.memory_summary(device, abbreviated=True))



def train(model, n, c, x, edge_index, num_batch, train_mask, true_label, criterion, optimizer, device, args, x_agg=None):
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
        # x_agg_i = x_agg[idx_i].to(device)
        edge_index_i, _ = subgraph(idx_i, edge_index, relabel_nodes=True, num_nodes=n)
        edge_index_i = edge_index_i.to(device)
        # adj_t_i = edge_index_to_adj_t(edge_index_i, x_i.shape[0])
        # Validate edge indices
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
            out_i = model(x_i, edge_index_i)
            
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            print(f"x_i shape: {x_i.shape}, edge_index_i shape: {edge_index_i.shape}")
            exit()  # Skip this batch
        if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
            loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i].to(torch.float))

        else:
            out_i = F.log_softmax(out_i, dim=1)
            loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i])
       
        loss.backward()
        optimizer.step()
        
        
        
    return 0, loss

def train_large(model, train_loader, criterion, optimizer, device, args, x_agg=None):
    print("Inside Train neighbor")
    model.train()
    # model.to(device)
    total_loss = 0
    for i, batch in enumerate(train_loader):
        # print(i)
        batch = batch.to(device)
        batch.y = batch.y.to(torch.long)
        
        optimizer.zero_grad()
        
        if hasattr(batch, 'adj_t') and batch.adj_t is not None:
            out = model(batch.x, batch.adj_t)
        else:
            # raise ValueError("error adj_t not found")
            out = model(batch.x, batch.edge_index)
        
        
        if(args.dataset == "ogbn-papers100M"):
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size].squeeze(1))
        elif(args.dataset == "ogbn-products"):
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, batch.y.squeeze(1))
        
    
        loss.backward()
        optimizer.step()
        if(args.dataset == "ogbn-papers100M"):
             total_loss += loss.item() * len(batch.input_id)
        else:
             total_loss += loss.item() * batch.x.shape[0]

        del batch, loss, out  # Free computation graph
        

    return 0, total_loss / len(train_loader.dataset)


@torch.no_grad()
def test_large(model, device, train_loader, val_loader, test_loader, x_agg, evaluator, args, eval_func):
    model.eval()
    result = []
    # Helper function to evaluate a single loader
    def evaluate_loader(loader):
        total_correct = 0
        total_samples = 0
        
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            batch.y = batch.y.to(torch.long)
            # x_agg_batch = x_agg[batch.n_id.cpu()].to(device, non_blocking=True)
            # out = model(batch.x, batch.edge_index)
            if hasattr(batch, 'adj_t') and batch.adj_t is not None:
                out = model(batch.x, batch.adj_t)
            else:
                # raise ValueError("error adj_t not found")
                out = model(batch.x, batch.edge_index)

            if args.dataset == "ogbn-papers100M":
                pred = torch.argmax(out[:batch.batch_size], dim=1, keepdim=True).cpu()
                target = batch.y[:batch.batch_size].cpu()
            else:
                pred = torch.argmax(out, dim=1, keepdim=True).cpu()
                target = batch.y.cpu()
            
            total_correct += (pred == target).sum().item()
            total_samples += target.size(0)
            
            # Clear immediately
            del batch, out, pred, target
        torch.cuda.empty_cache()
        
        return total_correct / total_samples   
        
        
        

    result.append(evaluate_loader(train_loader))
    result.append(evaluate_loader(val_loader))
    result.append(evaluate_loader(test_loader))
    print(result)
    return result


def main(args):
    
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

    ### Basic information of datasets ###
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    # infer the number of classes for non one-hot and one-hot labels
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    # print(f"c: {c} and d: {d}")

    print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

    

    
    if not args.directed and args.dataset != 'ogbn-proteins':
        undir_time = timer()
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        print("Conversion to undirected took time: ", timer() - undir_time)
        needs_directed_model = False
    else:
        needs_directed_model = True

    
    if needs_directed_model:
        lap_edge_index = to_undirected(dataset.graph['edge_index'])
        lap_edge_index, _ = remove_self_loops(lap_edge_index)
       
        
    else:
        
        lap_edge_index, _ = remove_self_loops(dataset.graph['edge_index'])

    

    
    if needs_directed_model:
        
        dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
    else:
        model_edge_index, _ = add_self_loops(lap_edge_index, num_nodes=n)
        dataset.graph['edge_index'] = model_edge_index
    
    
    x_agg = None

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
    
    start_time=time.time()
    
    if(args.dataset in large_dataset_training):
        evaluator = Evaluator(name=args.dataset)
    
    # Converting edge_index to adj_t
    if(args.dataset in large_dataset_training):
        dataset._data.edge_index = dataset.graph['edge_index']
        dataset.convert_to_adj_t_after_preprocessing()
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
        # model.train()
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
            train_loader = NeighborLoader(dataset.pyg_data, input_nodes=split_idx['train'], shuffle=True, drop_last=True,num_workers=12,
                                        **kwargs)
            val_loader = NeighborLoader(dataset.pyg_data, input_nodes=split_idx['valid'], num_workers=12, **kwargs)
            test_loader = NeighborLoader(dataset.pyg_data, input_nodes=split_idx['test'], num_workers=12,  **kwargs)

            # del dataset
        
        else:
            train_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[split_idx['train']] = True
            # train_idx = split_idx['train'].to(device)
            num_batch = n // args.batch_size + (n % args.batch_size > 0)

      
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
            
            
            eval_start_time = timer() 
            if epoch !=0 and epoch % args.eval_step == 0:
                if(args.dataset in large_dataset_training):
                    result = test_large(model, device, train_loader, val_loader, test_loader, x_agg, evaluator, args, eval_func)
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
                    
                   
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'best_val': best_val,

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
                  
            eval_end_time = timer()

            prune_start_time = timer()
            
            prunt_end_time = timer()
            
            epoch_end = timer()
            total_epoch_time = total_epoch_time + (epoch_end - epoch_start)
            total_prune_time = total_prune_time + (prunt_end_time - prune_start_time)
            total_eval_time = total_eval_time + (eval_end_time - eval_start_time)
            
            if 'out' in locals() and 'loss' in locals():
                del out, loss
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
        
        final_result = [best_train, best_val, best_test, 0, 0]
        logger.add_result(run, final_result[:-1])
        logger.print_statistics(run)
        # Clear model and optimizer
        del model, optimizer
    logger.print_statistics()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
