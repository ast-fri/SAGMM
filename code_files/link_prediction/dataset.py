
import torch

import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset


def load_dataset(data_dir, dataname, sub_dataname=''):
    print(dataname)
    if dataname in ('ogbl-ddi', 'ogbl-ppa'):
        dataset = load_link_ogb_dataset(data_dir, dataname)
    elif dataname in ('ogbl-collab'):
        dataset = load_link_ogb_collab_dataset(data_dir, dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset

def load_link_ogb_dataset(data_dir, name):
    
    dataset = PygLinkPropPredDataset(name=name, transform=T.ToSparseTensor(), root=f'{data_dir}')
    
    def ogb_edge_idx_to_tensor():
            split_edge = dataset.get_edge_split()
            
            idx = torch.randperm(split_edge['train']['edge'].shape[0])
            idx = idx[:split_edge['valid']['edge'].shape[0]]
            split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
            tensor_split_idx = {
                outer_key: {inner_key: torch.as_tensor(split_edge[outer_key][inner_key])
                            for inner_key in split_edge[outer_key]}
                for outer_key in split_edge
            }
            return tensor_split_idx
    dataset.load_fixed_splits = ogb_edge_idx_to_tensor  # ogb_dataset.get_idx_split
    # dataset.label = torch.as_tensor(dataset.labels).reshape(-1, 1)
    print("Dataset: ", dataset[0])
    return dataset

def load_link_ogb_collab_dataset(data_dir, name):
    
    dataset = PygLinkPropPredDataset(name=name, root=f'{data_dir}')

    def ogb_edge_idx_to_tensor():
            split_edge = dataset.get_edge_split()
            tensor_split_idx = {
                outer_key: {inner_key: torch.as_tensor(split_edge[outer_key][inner_key])
                            for inner_key in split_edge[outer_key]}
                for outer_key in split_edge
            }
            return tensor_split_idx
    dataset.load_fixed_splits = ogb_edge_idx_to_tensor  # ogb_dataset.get_idx_split
   
    print("Dataset: ", dataset[0])
    return dataset

