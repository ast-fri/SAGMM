import argparse
import os
import random
import numpy as np
import torch

import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, negative_sampling

import matplotlib.pyplot as plt
from timeit import default_timer as timer
from logger import Logger
import warnings
from torch.utils.data import DataLoader
from ogb.linkproppred import Evaluator

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
from eval import evaluate_link
from parse import parse_method, parser_add_main_args
import time
from timeit import default_timer as timer
import gc
from torch_geometric.utils import to_undirected
from graph_positional_encoding import * 
warnings.filterwarnings('ignore')

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
    


def print_memory_summary(device):
    allocated = torch.cuda.memory_allocated(device) / 1e6
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e6
    print(f"Memory Allocated: {allocated:.2f} MB")
    print(f"Max Memory Allocated: {max_allocated:.2f} MB")
    print(torch.cuda.memory_summary(device, abbreviated=True))



def train(model, predictor, data, edge_index, split_idx, optimizer, batch_size, device, ep, x_agg, args):
    model.train()
    predictor.train()
    
    pos_train_edge = split_idx['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    
    total_batches = len(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True))
 
    for batch_idx, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True)):
        is_last_batch = (batch_idx == total_batches - 1)
        optimizer.zero_grad()
        if(args.method == 'SAGMM'):
            if(args.dataset == "ogbl-ppa"):
                out, aux_loss, _ = model(data.x, data.adj_t, x_agg=x_agg, last_idx=is_last_batch, adj_t_norm=data.adj_t_norm)
            else:
                out, aux_loss, _ = model(data.x, data.adj_t, x_agg=x_agg, last_idx=is_last_batch)
        else:
            if(args.dataset == "ogbl-ppa" and (args.method == "gcnwithjk" or args.method == "gcn")):
                out = model(data.x, data.adj_t_norm)
            else:
                out = model(data.x, data.adj_t)
            aux_loss = 0
            
        edge = pos_train_edge[perm].t()
        pos_out = predictor(out[edge[0]], out[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        if(args.dataset == 'ogbl-ddi'):
            edge_neg = negative_sampling(edge_index, num_nodes=data.x.size(0),
                                        num_neg_samples=perm.size(0), method='dense')
        else:
            edge_neg = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=out.device)
        neg_out = predictor(out[edge_neg[0]], out[edge_neg[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        if(args.add_aux_loss and args.method == 'SAGMM'):
            loss = pos_loss + neg_loss + aux_loss
        else:
            loss = pos_loss + neg_loss
        loss.backward()
    
        if(args.dataset == 'ogbl-ddi'):
            torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return 0, total_loss / total_examples


def main(args):
    if(args.epochs % args.prune_interval == 0):
        print("!!!! Warning, do not keep prune interval as multiple of epoch !!!! ")
    checkpoint_dir = f'{args.runs_path}/runs_{args.dataset}/results_{args.folder_name}/checkpoints'
    result_save_dir=f"{args.runs_path}/runs_{args.dataset}/results_{args.folder_name}"
    log_file_dir = f"{args.runs_path}/parameters/{args.dataset}/"
    lap_save_dir = f"../../lap_matrix/"
    
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
    if(args.method == "SAGMM"):
        set_random_seed(args.seed)
    else:
        set_random_seed(12345)

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
    else:
        data.x = data.x.to(torch.float)

    adj_t = data.adj_t
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
 
        
    # For ogbl citation it is not hits@K metric but to keep code structure is kept same so K=100
    if(args.dataset == 'ogbl-ddi'):
        K = 20
    elif(args.dataset == 'ogbl-collab'):
        K = 50   
    else:
        K = 100


    loggers = {
        f'Hits@{K}': Logger(args.runs, args)
    }
   
    
    with open(result_save_dir+'/results_best_val.txt', 'w') as f:
        f.write("Run\tEpoch\tLoss\tTrain\tValid\tTest\n")
    with open(result_save_dir+'/results_all.txt', 'w') as f:
        f.write("Run\tEpoch\tLoss\tTrain\tValid\tTest\tTime_Taken_So_Far (in Minutes)\n")

    start_time=time.time()

    val_acc_per_expert = {}  # Stores best validation accuracy per expert count
    test_acc_per_expert = {}  # Stores corresponding test accuracy
    train_acc_per_expert = {}
    evaluator = Evaluator(name=args.dataset)

    for run in range(args.runs):
       
        if(args.method == "SAGMM"):
            set_random_seed(args.seed+run)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
        
        train_losses = []
        valid_accuracies = []
        test_accuracies = []
        epoch_list=[]
        if(args.dataset == 'ogbl-ddi'):
            torch.nn.init.xavier_uniform_(emb.weight)
        
        model = parse_method(args, 0, d, n, device)
        predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                                    args.gnn_num_layers, args.gnn_dropout).to(device)
        
        if(run == 0):
            print('MODEL:', model)
        print()
        print(f"************ Run{run} *************")
        
        # Get split for that run
        split_idx = split_idx_lst[run]
        
        
        if(args.dataset == 'ogbl-ddi'):
            optimizer = torch.optim.Adam(list(model.parameters()) + list(emb.parameters()) + list(predictor.parameters()), weight_decay=args.gnn_weight_decay, lr=args.lr)
        else:
            optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)

        # Initialize 
        if(args.method == "SAGMM"):
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
                out, loss = train(model, predictor, data, edge_index, split_idx, optimizer, args.batch_size, device, ep=None, x_agg=x_agg, args=args)
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("Caught OOM error!")
                    print_memory_summary(device)
                print(e)
                exit() 
            
            # Track model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # Save expert_outputs as a PyTorch tensor
            
            with open(log_file, "a") as file: 
                file.write(f"{epoch}\t{total_params}\t{trainable_params}\n")
            
            # Evaluation stage
            eval_start_time = timer()  
            if (epoch) % args.eval_step == 0:
                result = evaluate_link(model, predictor, data, split_idx, evaluator, args.batch_size, args.method, device, ep=None, x_agg=x_agg, K=K, dataset=args.dataset)
                for key, result in result.items():
                        train_hits, valid_hits, test_hits = result
                
                train_losses.append(loss)
                valid_accuracies.append(valid_hits)
                current_val_accuracy = valid_hits
                test_accuracies.append(test_hits)
                epoch_list.append(epoch)
                end_time=time.time()
                # Save intermediate results
                with open(result_save_dir + '/results_all.txt', 'a') as f:
                    time_so_far = (time.time() - start_time) / 60  # in minutes
                    f.write(f"{run}\t{epoch}\t{loss:.4f}\t{train_hits:.4f}\t{valid_hits:.4f}\t{test_hits:.4f}\t{time_so_far:.2f}\n")
                
                if best_val == 0:
                    best_val = valid_hits
                if valid_hits >= best_val:
                    best_val = valid_hits
                    best_test = test_hits
                    best_train = train_hits
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{run}.pt')
                    expert_state_dicts = {}
                    # print("test accuracy: ", test_hits)
                    if(args.method == "SAGMM"): 
                        for i in range(args.max_experts):
                            if(gmoe_network.experts[i] is not None):
                                expert_state_dicts[i] = gmoe_network.experts[i].state_dict()
                            
                        
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'predictor':predictor.state_dict(),
                            'loss': loss,
                            'best_val': best_val,
                            'embedding': out,
                            'remaining_experts': gmoe_network.num_experts,
                            'expert_gate_indices': gmoe_network.expert_gate_indices,  # Keep track of active experts
                            'experts': gmoe_network.experts,
                            'experts_names': gmoe_network.expert_names,
                            'expert_state_dict': expert_state_dicts,
                            }, checkpoint_path)
                    else:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'predictor':predictor.state_dict(),
                            'loss': loss,
                            'best_val': best_val,
                            'embedding': out,
                            }, checkpoint_path)
                    # Save best results
                    with open(result_save_dir + '/results_best_val.txt', 'a') as f:
                        f.write(f"{run}\t{epoch}\t{loss:.4f}\t{train_hits:.4f}\t{valid_hits:.4f}\t{test_hits:.4f}\n")
                
                if epoch % args.display_step == 0:
                    print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                    # print()
            eval_end_time = timer()

            prune_start_time = timer()
            if (not args.stop_prune and ((epoch + 1) % args.prune_interval == 0) and args.method == "SAGMM"):
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

                print(f"Importance values: {valid_importance.tolist()}")
                print(f"Threshold: {threshold:.4f}")
                print(f"Num experts to remove: {num_experts_to_remove}")

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
        
        final_result = [best_train, best_val, best_test, 0, 0]
        loggers[f'Hits@{K}'].add_result(run, final_result[:-1])
        loggers[f'Hits@{K}'].print_statistics(args.dataset,args.epochs,args.hidden_channels,args.gnn_dropout,args.coef,args.k,args.seed,run)
        
    loggers[f'Hits@{K}'].print_statistics(args.dataset,args.epochs,args.hidden_channels,args.gnn_dropout,args.coef,args.k,args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Pipeline for Link Prediction')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)

