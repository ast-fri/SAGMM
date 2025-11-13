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
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from eval import evaluate, evaluate_large
from parse import parse_method, parser_add_main_args
from scipy.stats import pearsonr, spearmanr
from scipy.stats import entropy
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from timeit import default_timer as timer
from graph_positional_encoding import * 
from utils import preprocess_dataset, preprocess_dataset_lap
warnings.filterwarnings('ignore')
## loss functions
cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, device, loader, optimizer, task_type, x_agg=None):
    model.train()
    bs=0
    for batch in loader:
        bs+=1
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred, aux_loss, raw_logits = model(batch, x_agg)
            optimizer.zero_grad()
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss =loss + aux_loss
            loss.backward()
            optimizer.step()
    return loss

def eval(model, device, loader, evaluator, x_agg=None):
    model.eval()
    y_true = []
    y_pred = []
    for batch in loader:
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred,_,_ = model(batch,x_agg)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)
def main():
    ### initialization for plotting
    best_val = float('-inf')
    ### Parse args ###
    parser = argparse.ArgumentParser(description='Training Pipeline for Graph Classification or Regression')
    parser_add_main_args(parser)
    # Add arguments for expert management
    args = parser.parse_args()
    print(args)
    checkpoint_dir = f'{args.runs_path}/runs_{args.dataset}/results_{args.folder_name}/checkpoints'
    result_save_dir=f"{args.runs_path}/runs_{args.dataset}/results_{args.folder_name}"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_save_dir, exist_ok=True)
    
    set_random_seed(args.seed)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        print("device: ",device)
    ### Load and preprocess data ###
    dataset = PygGraphPropPredDataset(name = args.dataset, root= args.data_dir)
    print("#Number of graphs: ", len(dataset))
    split_idx_lst = [dataset.get_idx_split() for _ in range(args.runs)] 
    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    evaluator = Evaluator(args.dataset)
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()
    logger = Logger(args.runs, args)
    with open(result_save_dir+'/results_best_val.txt', 'w') as f:
        f.write("Run\tEpoch\tLoss\tTrain\tValid\tTest\n")
    with open(result_save_dir+'/results_all.txt', 'w') as f:
        f.write("Run\tEpoch\tLoss\tTrain\tValid\tTest\tTime_Taken_So_Far (in Minutes)\n")

    start_time=time.time()
    ### Training loop ###
    expert_usage = []
    gate_values = []
    val_acc_per_expert = {}  # Stores best validation accuracy per expert count
    test_acc_per_expert = {}  # Stores corresponding test accuracy
    train_acc_per_expert = {}
    lap_save_dir = f"preprocess_outputs/"
    os.makedirs(lap_save_dir, exist_ok=True)
    if(args.encode_mode == "lap"):
        evals, evects = None, None
        args.max_freqs = 10
        lap_save_path = f"{lap_save_dir}/{args.dataset}_{args.max_freqs}_lap_matrix.pt"
        if os.path.exists(lap_save_path):
            print(f"Loading precomputed Laplacian from {lap_save_path}...")
            data = torch.load(lap_save_path)
            evals = data['evals']
            evects = data['evects']
        else:
            with torch.no_grad():
                print("Computing Laplacian eigen-decomposition...")
                start = timer()
                lap_pos = AddLaplacianPE(args.max_freqs, is_undirected=True)
                evects, evals = lap_pos(dataset.graph)
                end = timer()
                # Save results for future use
                print(f"Saving Laplacian eigen-decomposition to {lap_save_path}...")
                print("Laplacian formation time: ", end -start)

        lap_pe = evects.to(dataset.graph["node_feat"].device)
        laplacian_encoder = torch.nn.Linear(args.max_freqs, (dataset.graph["node_feat"]).shape[1]).to(device)
        lap_pe_proj = laplacian_encoder(lap_pe)
        if(args.agg_mode == "add"):
            x_agg = torch.add(0.3* dataset.graph['node_feat'], 0.7 * lap_pe_proj)
        elif(args.agg_mode == "concat"):
            x_agg = torch.cat([dataset.graph['node_feat'], lap_pe_proj], dim=1)
        else:
            x_agg = lap_pe_proj
        x_agg = x_agg.detach()
    elif (args.encode_mode == "multihop_lap"):
            evals, evects = None, None
            lap_save_path = f"{lap_save_dir}/{args.dataset}_{args.max_freqs}_lap_matrix_and_multihop_dataset.pt"
            with torch.no_grad():
                start = timer()
                ## append x_agg directly to the data, preprocessingg time: ~6 minutes for molhiv dataset
                dataset = preprocess_dataset_lap(dataset, aggregation_type="local", save_path = lap_save_path, device=device, batch_size=args.batch_size, max_freq_lap=args.max_freqs)
                print(dataset)
                print(dataset[0])
                end = timer()
                print("Total preprocessing time (Multihop + Laplacian): ", end - start)
    elif (args.encode_mode == "graph_mean_lap"):
            evals, evects = None, None
            lap_save_path = f"{lap_save_dir}/{args.dataset}_{args.max_freqs}_lap_matrix_and_global_dataset.pt"
            with torch.no_grad():
                start = timer()
                dataset = preprocess_dataset_lap(dataset, aggregation_type="global", save_path = lap_save_path, device=device, batch_size=args.batch_size, max_freq_lap=args.max_freqs)
                print(dataset)
                print(dataset[0])
                end = timer()
                print("Total preprocessing time (Global Mean + Laplacian): ", end - start)
    elif (args.encode_mode == "graph_mean"):
            evals, evects = None, None
            lap_save_path = f"{lap_save_dir}/{args.dataset}_global_mean_into_feats_dataset.pt"
            with torch.no_grad():
                start = timer()
                dataset = preprocess_dataset(dataset, aggregation_type="global", save_path = lap_save_path, device=device, batch_size=args.batch_size)
                print(dataset)
                print(dataset[0])
                end = timer()
                print("Total preprocessing time (Global Mean): ", end - start)
    elif (args.encode_mode == "multihop"):
            evals, evects = None, None
            lap_save_path = f"{lap_save_dir}/{args.dataset}_multihop_mean_into_feats_dataset.pt"
            with torch.no_grad():
                start = timer()
                dataset = preprocess_dataset(dataset, aggregation_type="local", save_path = lap_save_path, device=device, batch_size=args.batch_size)
                print(dataset)
                print(dataset[0])
                end = timer()
                print("Total preprocessing time (MultiHop): ", end - start)

    elif (args.encode_mode == "none"):
        print("Warning!! proceeding without encode mode")
    for run in range(args.runs):
        set_random_seed(args.seed+run)
        model = parse_method(args, dataset, device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if(run == 0):
            print('MODEL:', model)
        print()
        print(f"************ Run{run} *************")
        train_losses = []
        valid_accuracies = []
        test_accuracies = []
        epoch_list=[]
        split_idx = split_idx_lst[run]
        # train_idx = split_idx['train'].to(device)
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        # Initialize gradient norms accumulator for the GMoE_network
        if args.expert_type == "multiple_models":
            gmoe_network = model.network_model.experts[0]
        best_val = 0
        best_test = 0
        best_train = 0
        current_val_accuracy = 0
        total_epoch_time = 0
        total_prune_time = 0
        total_eval_time = 0
        for epoch in range(args.epochs):
            # print("epoch: ",epoch)
            epoch_start = timer()
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
            # print("Gmoe num experts: ", gmoe_network.num_experts)
            loss = train(model, device, train_loader, optimizer, dataset.task_type)
            if args.expert_type == "multiple_models":
                eval_start_time = timer()  
                if epoch % args.eval_step == 0:
                    train_perf = eval(model, device, train_loader, evaluator)
                    valid_perf = eval(model, device, valid_loader, evaluator)
                    test_perf = eval(model, device, test_loader, evaluator)
                    train_losses.append(loss.item())
                    valid_accuracies.append(valid_perf[dataset.eval_metric])
                    current_val_accuracy = valid_perf[dataset.eval_metric]
                    test_accuracies.append(test_perf[dataset.eval_metric])
                    epoch_list.append(epoch)
                    end_time=time.time()
                    with open(result_save_dir+'/results_all.txt', 'a') as f:
                        f.write(f"{run}\t{epoch}\t{loss.item():.4f}\t{train_perf[dataset.eval_metric]:.4f}\t{valid_perf[dataset.eval_metric]:.4f}\t{test_perf[dataset.eval_metric]:.4f}\t\t{(end_time-start_time)/60:.2f}\n")
                    # Save checkpoint if validation accuracy improved
                    if valid_perf[dataset.eval_metric] > best_val:
                        best_val = valid_perf[dataset.eval_metric]
                        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{run}.pt')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            'best_val': best_val
                        }, checkpoint_path)
                        # Save results in text file
                        with open(result_save_dir+'/results_best_val.txt', 'a') as f:
                            f.write(f"{run}\t{epoch}\t{loss.item():.4f}\t{train_perf[dataset.eval_metric]:.4f}\t{valid_perf[dataset.eval_metric]:.4f}\t{test_perf[dataset.eval_metric]:.4f}\n")
                    if "classification" in dataset.task_type:
                        print_str = f'Epoch: {epoch:02d}, ' + \
                                    f'Loss: {loss.item():.4f}, ' + \
                                    f'Train: {100 * train_perf[dataset.eval_metric]:.2f}%, ' + \
                                    f'Valid: {100 * valid_perf[dataset.eval_metric]:.2f}%, ' + \
                                    f'Test: {100 * test_perf[dataset.eval_metric]:.2f}%'
                        print(print_str)
                    else:
                        # print(f"Epoch: ")
                        print_str = f'Epoch: {epoch:02d}, ' + \
                                    f'Loss: {loss.item():.4f}, ' + \
                                    f'Train RMSE: {train_perf[dataset.eval_metric]:.2f}, ' + \
                                    f'Valid RMSE: {valid_perf[dataset.eval_metric]:.2f}, ' + \
                                    f'Test RMSE: {test_perf[dataset.eval_metric]:.2f}'
                        print(print_str)
                eval_end_time = timer()
                prune_start_time = timer()
                
                if (epoch + 1) % args.prune_interval == 0:
                    print(f"Inside expert removal process")

                    # Compute the importance metrics and filter valid contributions
                    valid_mask = (gmoe_network.ema_contributions != float('-inf'))
                    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                    valid_importance = (gmoe_network.ema_contributions / len(dataset))[valid_indices]
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
                # print()   
            
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

if __name__ == "__main__":
    main()