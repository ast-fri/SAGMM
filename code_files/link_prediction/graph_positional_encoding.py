from typing import Any, Optional
import numpy as np
import torch
from torch import Tensor
import torch_geometric.typing
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
    to_networkx
)
import torch.nn.functional as F
from collections import deque
from torch import svd_lowrank
import scipy.sparse as sp
import scipy.sparse
import numpy.testing as npt
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter)

import torch
import torch.nn as nn
from timeit import default_timer as timer
import torch
import torch_geometric as pyg
from torch_geometric.utils import degree, to_dense_adj, remove_self_loops, add_self_loops
from torch_scatter import scatter_add
from torch_sparse import SparseTensor
import torch.nn.functional as F

from torch_scatter import scatter_mean
import os
import torch

EPS = 1e-5

import torch
import numpy as np

EPS = 1e-5


@functional_transform('add_laplacian_pe')
class AddLaplacianPE(BaseTransform):
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    # Number of nodes from which to use sparse eigenvector computation:
    SPARSE_THRESHOLD: int = 100

    def __init__(
        self,
        k: int,
        attr_name: Optional[str] = 'laplacian_eigenvector_pe',
        is_undirected: bool = False,
        **kwargs: Any,
    ) -> None:
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def forward(self, data, lap_edge_index) -> Data:
        assert lap_edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        edge_index, edge_weight = get_laplacian(
            edge_index=lap_edge_index,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        if num_nodes < self.SPARSE_THRESHOLD:
            from numpy.linalg import eig, eigh
            eig_fn = eig if not self.is_undirected else eigh

            eig_vals, eig_vecs = eig_fn(L.todense())  # type: ignore
        else:
            from scipy.sparse.linalg import eigs, eigsh
            eig_fn = eigs if not self.is_undirected else eigsh

            eig_vals, eig_vecs = eig_fn(  # type: ignore
                L,
                k=self.k + 1,
                which='SR' if not self.is_undirected else 'SA',
                return_eigenvectors=True,
                **self.kwargs,
            )

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        pe *= sign

        # data = add_node_attr(data, pe, attr_name=self.attr_name)
        return pe, eig_vals[1:self.k + 1]



def scale_feature(feature, node_feat_mean):
    """Scale structural feature based on node features mean"""
    feature_mean = torch.mean(torch.abs(feature))
    if feature_mean == 0:
        return feature
    scale_factor = node_feat_mean / (feature_mean + 1e-8)
    return feature * scale_factor


# mark

def aggregate_hops_mean(x, edge_index,device, args):
    ## assumes no duplicates in edge index
    # Ensure the graph is undirected by adding reverse edges and removing duplicates
    # src, dst = edge_index
    # src_both = torch.cat([src, dst])
    # dst_both = torch.cat([dst, src])
    # edge_index_undirected = torch.unique(torch.stack([src_both, dst_both]), dim=1)  # Remove duplicates
    
    src, dst = edge_index
    src = src.to(device)
    dst = dst.to(device)
    # 1-hop aggregation
    x_1hop = scatter_mean(x[src], dst, dim=0, dim_size=x.size(0))

    if(args.dataset == "ogbn-papers100M"):
        return x_1hop
    # # 2-hop aggregation using adjacency propagation
    x_2hop = scatter_mean(x_1hop[src], dst, dim=0, dim_size=x.size(0))

    # Preserve self-information and prevent zero features for isolated nodes
    # x_hop_agg = (x_1hop + x_2hop) / 2
    # x_hop_agg = (x + x_1hop) / 2

    return x_1hop, x_2hop


from torch_geometric.nn import Node2Vec
def save_embedding(model, save_path):
    torch.save(model.embedding.weight.data.cpu(), save_path)

def normalize_feature(feature, method='z-score', node_feat_mean = None):
    """Normalize features using either min-max, z-score, or log scaling."""
    if method == 'min-max':
        return (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
    elif method == 'z-score':
        return (feature - feature.mean()) / (feature.std() + 1e-8)
    elif method == 'log':
        return torch.log1p(feature)  # log(1 + x) to handle skewed distributions
    elif method == 'mean_scale':
        feature_mean = torch.mean(torch.abs(feature))
        if feature_mean == 0:
            return feature
        scale_factor = node_feat_mean / (feature_mean + 1e-8)
        return feature * scale_factor
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
def get_graph_info(args, lap_save_dir, dataset, d, undir_edge_index, x, edge_index, device):
    
    node_mean_abs = torch.mean(torch.abs(x))
    if(args.encode_mode == "lap"):
        evals, evects = None, None
        if(args.dataset == "deezer-europe"):
            args.max_freqs = 10
        elif(args.dataset == "ogbn-proteins"): 
            args.max_freqs = 6
        elif(args.dataset == "ogbn-papers100M"): 
            args.max_freqs = 20
        else:
            args.max_freqs = d
        lap_save_path = f"{lap_save_dir}/{args.dataset}_{args.max_freqs}_lap_matrix.pt"

        if os.path.exists(lap_save_path):
            print(f"Loading precomputed Laplacian from {lap_save_path}...")
            data = torch.load(lap_save_path, map_location=device)
            evals = data['evals']
            evects = data['evects']

        else:
            with torch.no_grad():
                print("Computing Laplacian eigen-decomposition...")
                start = timer()
                lap_pos = AddLaplacianPE(args.max_freqs, is_undirected=True)
                evects, evals = lap_pos(dataset.graph, undir_edge_index)
                torch.save({'evals': torch.tensor(evals), 'evects': torch.tensor(evects)}, lap_save_path)
               
                end = timer()
                # Save results for future use
                print(f"Saving Laplacian eigen-decomposition to {lap_save_path}...")
                
                torch.save({'evals': torch.tensor(evals), 'evects': torch.tensor(evects)}, lap_save_path)
                print("LAP formation time: ", end -start)
        
        
        lap_pe = evects.to(x.device)
        normalized_eigvecs = scale_feature(lap_pe, node_mean_abs)
        if(args.agg_mode == "add"):
            x_agg = torch.add(dataset.graph['node_feat'], lap_pe)
        elif(args.agg_mode == "concat"):
            if(args.dataset == "ogbl-ddi"):
                x_agg = lap_pe
                args.max_freqs = 0
            # x_agg = torch.cat([x, lap_pe], dim=1)
            else:
                x_agg = torch.cat([data.x, normalized_eigvecs], dim=1)
        else:
            x_agg = x
    
    elif (args.encode_mode == "multihop_lap"):
        evals, evects = None, None
        if(args.dataset == "ogbl-collab"):
            args.max_freqs = 10
        elif(args.dataset == "ogbl-ppa"):
            args.max_freqs = 15
        else:
            args.max_freqs = d
        lap_save_path = f"{lap_save_dir}/{args.dataset}_{args.max_freqs}_lap_matrix.pt"
        multihop_save_path = f"{lap_save_dir}/{args.dataset}_multihop_features.pt"
        
        if (os.path.exists(multihop_save_path) and os.path.exists(lap_save_path)):
            print(f"Loading precomputed multihop features from {multihop_save_path} and laplacian from {lap_save_path}...")
            multihop_features = torch.load(multihop_save_path, map_location=device)
            data = torch.load(lap_save_path, map_location=device)
            if(args.dataset == "ogbn-papers100M"):
                x_1hop = multihop_features['x_1hop']
            else:
                x_1hop = multihop_features['x_1hop']
                x_2hop = multihop_features['x_2hop']
            evals = data['evals']
            evects = data['evects']
        else:
            with torch.no_grad():
                start = timer()
                
                if(os.path.exists(lap_save_path)):
                    data = torch.load(lap_save_path, map_location=device)
                    evals = data['evals']
                    evects = data['evects']
                
                else:
                    lap_pos = AddLaplacianPE(args.max_freqs, is_undirected=True)
                    evects, evals = lap_pos(dataset, undir_edge_index)
                    torch.save({'evals': torch.tensor(evals), 'evects': torch.tensor(evects)}, lap_save_path)
                
                if(os.path.exists(multihop_save_path)):
                    multihop_features = torch.load(multihop_save_path, map_location=device)
                    if(args.dataset == "ogbn-papers100M"):
                        x_1hop = multihop_features['x_1hop']
                    else:
                        x_1hop = multihop_features['x_1hop']
                        x_2hop = multihop_features['x_2hop']
                else:
                    if(args.dataset == "ogbn-papers100M"):
                        x_1hop= aggregate_hops_mean(x, edge_index, device, args)
                        # Save the computed features
                        multihop_features = {
                            'x_1hop': x_1hop,
                        }
                    else:
                        x_1hop, x_2hop = aggregate_hops_mean(x, edge_index, device, args)
                        # Save the computed features
                        multihop_features = {
                            'x_1hop': x_1hop,
                            'x_2hop': x_2hop,
                        }
                
                    torch.save(multihop_features, multihop_save_path)
                end = timer()
                # Save results for future use
                # print(f"Saving Laplacian eigen-decomposition to {lap_save_path}...")
                
                
                print("Multihop features calculation time: ", end - start)
        lap_pe = evects.to(x.device)
       
        if(args.dataset == "ogbn-papers100M"):
            x_1hop = x_1hop.to(x.device)
            x_agg = (x +  x_1hop) / 2
            x_agg_mean_abs = torch.mean(torch.abs(x_agg))
            normalized_eigvecs = scale_feature(lap_pe, x_agg_mean_abs)
            x_agg = torch.cat([x_agg, normalized_eigvecs], dim=1)
        else:
            x_1hop = x_1hop.to(x.device)
            x_2hop = x_2hop.to(x.device)
            x_agg = (x +  x_1hop + x_2hop) / 3
            x_agg_mean_abs = torch.mean(torch.abs(x_agg))
            normalized_eigvecs = scale_feature(lap_pe, x_agg_mean_abs)
            x_agg = torch.cat([x_agg, normalized_eigvecs], dim=1)

    elif (args.encode_mode == "none"):
        x_agg = x
        print("Warning!! proceeding without encode mode")
    
    return x_agg