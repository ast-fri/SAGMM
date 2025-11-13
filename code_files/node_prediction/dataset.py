
import numpy as np
import torch
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T
from data_utils import rand_train_test_idx, to_sparse_tensor, class_rand_splits
from torch_geometric.datasets import Planetoid
from os import path

from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset
import os
from torch_geometric.utils import subgraph


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """
        
        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))

    

def load_dataset(data_dir, dataname, sub_dataname=''):
    """ Loader for NCDataset 
        Returns NCDataset 
    """    
    if dataname == 'ogbn-proteins':
        dataset = load_proteins_dataset(data_dir)
    elif dataname == 'deezer-europe':
        dataset = load_deezer_dataset(data_dir)
    elif dataname == 'pokec':
        dataset = load_pokec_mat(data_dir)
    elif dataname == 'yelp-chi':
        dataset = load_yelpchi_dataset(data_dir)
    elif dataname in ('ogbn-arxiv'):
        dataset = load_ogb_dataset(data_dir, dataname)
    elif dataname == 'ogbn-products':
        dataset = load_ogb_products(data_dir)
    elif dataname == 'ogbn-papers100M':
        dataset = load_papers100M(data_dir)
    elif dataname in  ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(data_dir, dataname)
    
    else:
        raise ValueError('Invalid dataset name')
    return dataset


def load_deezer_dataset(data_dir):
    
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{data_dir}/deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset

def load_papers100M(data_dir):
    """Use OGB dataset directly without intermediate conversion"""
    ogb_dataset = PygNodePropPredDataset('ogbn-papers100M', root=data_dir)
    split_idx = ogb_dataset.get_idx_split()
    
    # Create a lightweight wrapper that provides both interfaces
    class OptimizedDataset:
        def __init__(self, ogb_dataset, split_idx):
            self.ogb_dataset = ogb_dataset
            self.split_idx = split_idx
            self.name = 'ogbn-papers100M'
            self._data = ogb_dataset[0]
            self.label = torch.as_tensor(self._data.y.data, dtype=int).reshape(-1, 1)
        
        def convert_to_adj_t_after_preprocessing(self):
            """Call this after all preprocessing is complete"""
            # if not self._preprocessing_done:
            from torch_sparse import SparseTensor
            
            row, col = self._data.edge_index
            self._data.adj_t = SparseTensor(
                row=row, col=col, 
                sparse_sizes=(self._data.num_nodes, self._data.num_nodes)
            )
        
            # Free the edge_index memory
            del self._data.edge_index
            self._data.edge_index = None
            
            self._preprocessing_done = True
        

        @property
        def pyg_data(self):
            return self._data  # Direct access, no copying
            
        @property 
        def graph(self):
            # data = self.ogb_dataset[0]
            return {
                'edge_index': self._data.edge_index,
                'node_feat': self._data.x,
                'num_nodes': self._data.num_nodes,
                'edge_feat': getattr(self._data, 'edge_attr', None)
            }
            
       
        def load_fixed_splits(self):
            return self.split_idx
    
    return OptimizedDataset(ogb_dataset, split_idx)

def load_proteins_dataset(data_dir):
    ogb_dataset = NodePropPredDataset(name='ogbn-proteins', root=f'{data_dir}')
    dataset = NCDataset('ogbn-proteins')
    def protein_orig_split(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        return {'train': torch.as_tensor(split_idx['train']),
                'valid': torch.as_tensor(split_idx['valid']),
                'test': torch.as_tensor(split_idx['test'])}
    dataset.load_fixed_splits = protein_orig_split
    dataset.graph, dataset.label = ogb_dataset.graph, ogb_dataset.labels

    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['edge_feat'] = torch.as_tensor(dataset.graph['edge_feat'])
    dataset.label = torch.as_tensor(dataset.label)

    edge_index_ = to_sparse_tensor(dataset.graph['edge_index'],
                                   dataset.graph['edge_feat'], dataset.graph['num_nodes'])
    dataset.graph['node_feat'] = edge_index_.mean(dim=1)
    dataset.graph['edge_feat'] = None

    return dataset

def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx
    dataset.load_fixed_splits = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset

def load_ogb_products(data_dir):
    """Use OGB dataset directly without intermediate conversion"""
    ogb_dataset = PygNodePropPredDataset('ogbn-products', root=data_dir)
    split_idx = ogb_dataset.get_idx_split()
    
    # Create a lightweight wrapper that provides both interfaces
    class OptimizedDataset:
        def __init__(self, ogb_dataset, split_idx):
            self.ogb_dataset = ogb_dataset
            self.split_idx = split_idx
            self.name = 'ogbn-products'
            self._data = ogb_dataset[0]
            self.label = torch.as_tensor(self._data.y.data).reshape(-1, 1)
        
        def convert_to_adj_t_after_preprocessing(self):
            """Call this after all preprocessing is complete"""
            # if not self._preprocessing_done:
            from torch_sparse import SparseTensor
            
            row, col = self._data.edge_index
            self._data.adj_t = SparseTensor(
                row=row, col=col, 
                sparse_sizes=(self._data.num_nodes, self._data.num_nodes)
            )
            
            # Free the edge_index memory
            del self._data.edge_index
            self._data.edge_index = None
            
            self._preprocessing_done = True
            print(f"Converted to adj_t and freed edge_index memory")

        @property
        def pyg_data(self):
            return self._data  # Direct access, no copying
            
        @property 
        def graph(self):
            data = self.ogb_dataset[0]
            return {
                'edge_index': data.edge_index,
                'node_feat': data.x,
                'num_nodes': data.num_nodes,
                'edge_feat': getattr(data, 'edge_attr', None)
            }
       
            
        def load_fixed_splits(self):
            return self.split_idx
    
    return OptimizedDataset(ogb_dataset, split_idx)

def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    
    # try:
    fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')
    edge_index = fulldata['edge_index']
    node_feat = fulldata['node_feat']
    label = fulldata['label']
    

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat).float()
    num_nodes = int(node_feat.shape[0])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = torch.tensor(label).flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    def load_fixed_splits(train_prop=0.5, val_prop=0.25):
        dir = f'{data_dir}pokec/split_0.5_0.25'
        tensor_split_idx = {}
        if os.path.exists(dir):
            tensor_split_idx['train'] = torch.as_tensor(np.loadtxt(dir + '/pokec_train.txt'), dtype=torch.long)
            tensor_split_idx['valid'] = torch.as_tensor(np.loadtxt(dir + '/pokec_valid.txt'), dtype=torch.long)
            tensor_split_idx['test'] = torch.as_tensor(np.loadtxt(dir + '/pokec_test.txt'), dtype=torch.long)
        else:
            os.makedirs(dir)
            tensor_split_idx['train'], tensor_split_idx['valid'], tensor_split_idx['test'] \
                = rand_train_test_idx(dataset.label, train_prop=train_prop, valid_prop=val_prop)
            np.savetxt(dir + '/pokec_train.txt', tensor_split_idx['train'], fmt='%d')
            np.savetxt(dir + '/pokec_valid.txt', tensor_split_idx['valid'], fmt='%d')
            np.savetxt(dir + '/pokec_test.txt', tensor_split_idx['test'], fmt='%d')
        return tensor_split_idx

    dataset.load_fixed_splits = load_fixed_splits
    return dataset

def load_yelpchi_dataset(data_dir):
    if not path.exists(f'{data_dir}YelpChi.mat'):
            print("!!!!!Yelpchi dataset not found!!!!!")
            gdd.download_file_from_google_drive(
                file_id= dataset_drive_url['yelp-chi'], \
                dest_path=f'{data_dir}YelpChi.mat', showsize=True)
    fulldata = scipy.io.loadmat(f'{data_dir}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset

def load_planetoid_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    torch_dataset = Planetoid(root=f'{data_dir}Planetoid',
                             name=name, transform=transform)
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset
