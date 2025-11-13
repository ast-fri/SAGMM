import os
# import dill
import torch
import argparse
import torch
from torch_scatter import scatter_mean
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from graph_positional_encoding import * 
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset

class PreprocessedGraphDataset(InMemoryDataset):
    def __init__(self, original_dataset, data_list, root=None, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        
        # Convert list of Data objects into a single Data object with batch info
        self.data, self.slices = self.collate(data_list)

        # Copy essential attributes from the original OGB dataset
        self.num_tasks = original_dataset.num_tasks
        self.eval_metric = original_dataset.eval_metric
        self.task_type = original_dataset.task_type
        self.get_idx_split = original_dataset.get_idx_split

    def __len__(self):
        return self.data.num_graphs
def scale_feature(feature, node_feat_mean):
    """Scale structural feature based on node features mean"""
    feature_mean = torch.mean(torch.abs(feature))
    if feature_mean == 0:
        return feature
    scale_factor = node_feat_mean / (feature_mean + 1e-8)
    return feature * scale_factor
def aggregate_hops_and_add_to_node(x, edge_index, device):
    """Aggregates 1-hop and 2-hop neighborhoods and returns updated features."""
    src, dst = edge_index
    src, dst = src.to(device), dst.to(device)

    x_1hop = scatter_mean(x[src], dst, dim=0, dim_size=x.size(0))
    x_2hop = scatter_mean(x_1hop[src], dst, dim=0, dim_size=x.size(0))

    # return (x + x_1hop + x_2hop)/3
    return x_1hop, x_2hop


def aggregate_global_mean_and_add(x, batch):
    """Computes the graph-level mean and returns updated features."""
    graph_mean = scatter_mean(x, batch, dim=0)
    # return (x + graph_mean[batch])/2
    return graph_mean


def preprocess_dataset(dataset, aggregation_type, device, save_path, batch_size):
    """
    Preprocesses node features by applying aggregation and Laplacian positional encoding.
    Updates the dataset with 'x_agg' and saves it. Loads from save_path if file exists.
    """
    # Load if preprocessed file exists
    if os.path.exists(save_path):
        print(f"Loading preprocessed dataset from {save_path}")
        data, slices = torch.load(save_path)
        dataset.data = data
        dataset.slices = slices
        return dataset

    print("Preprocessing and updating dataset...")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    x_agg_list = []
    # lap_pos = AddLaplacianPE(max_freq_lap, is_undirected=True)

    for data in tqdm(loader, desc="Processing graphs"):
        data = data.to(device)

        # Aggregation based on type
        if aggregation_type == "local":
            x_1hop, x_2hop= aggregate_hops_and_add_to_node(data.x, data.edge_index, device)
            x_preprocessed = (x_1hop + x_2hop + data.x)/3
        elif aggregation_type == "global":
            graph_mean = aggregate_global_mean_and_add(data.x, data.batch)
            x_preprocessed = (graph_mean[data.batch] + data.x)/2
        else:
            raise ValueError("Invalid aggregation type. Choose either 'local' or 'global'.")
        
        x_agg_list.append(x_preprocessed.cpu())

    # Update dataset with new x_agg feature
    x_agg_tensor = torch.cat(x_agg_list, dim=0)
    dataset.data.x_agg = x_agg_tensor
    dataset.slices['x_agg'] = dataset.slices['x'].clone()

    # Save the updated dataset
    torch.save((dataset.data, dataset.slices), save_path)
    print(f"Updated dataset saved at {save_path}")

    return dataset


def preprocess_dataset_lap(dataset, aggregation_type, device, save_path, batch_size, max_freq_lap=20):
    """
    Preprocesses node features by applying aggregation and Laplacian positional encoding.
    Updates the dataset with 'x_agg' and saves it. Loads from save_path if file exists.
    """
    # Load if preprocessed file exists
    if os.path.exists(save_path):
        print(f"Loading preprocessed dataset from {save_path}")
        data, slices = torch.load(save_path)
        dataset.data = data
        dataset.slices = slices
        return dataset

    print("Preprocessing and updating dataset...")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    x_agg_list = []
    lap_pos = AddLaplacianPE(max_freq_lap, is_undirected=True)

    for data in tqdm(loader, desc="Processing graphs"):
        data = data.to(device)
        # Aggregation based on type
        if aggregation_type == "local":
            x_1hop, x_2hop = aggregate_hops_and_add_to_node(data.x, data.edge_index, device)
            x_preprocessed = (x_1hop + x_2hop + data.x)/3
        elif aggregation_type == "global":
            graph_mean = aggregate_global_mean_and_add(data.x, data.batch)
            x_preprocessed = (graph_mean[data.batch] + data.x)/2
        else:
            raise ValueError("Invalid aggregation type. Choose either 'local' or 'global'.")
        # Add Laplacian positional encoding
        evects, _ = lap_pos(data)
        lap_pe = evects.to(device)

        x_agg_mean_abs = torch.mean(torch.abs(x_preprocessed))
        normalized_eigvecs = scale_feature(lap_pe, x_agg_mean_abs)

        x_agg = torch.cat([x_preprocessed, normalized_eigvecs], dim=1)
        x_agg_list.append(x_agg.cpu())

    # Update dataset with new x_agg feature
    x_agg_tensor = torch.cat(x_agg_list, dim=0)
    dataset.data.x_agg = x_agg_tensor
    dataset.slices['x_agg'] = dataset.slices['x'].clone()

    # Save the updated dataset
    torch.save((dataset.data, dataset.slices), save_path)
    print(f"Updated dataset saved at {save_path}")

    return dataset