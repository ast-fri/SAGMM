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

import torch.nn as nn
from timeit import default_timer as timer
import torch_geometric as pyg
from torch_geometric.utils import degree, to_dense_adj, remove_self_loops, add_self_loops
from torch_scatter import scatter_add
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean
EPS = 1e-5
def add_node_attr(
    data: Data,
    value: Any,
    attr_name: Optional[str] = None,
) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


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

    def forward(self, data: Data) -> Data:
        assert data['edge_index'] is not None
        num_nodes = data['num_nodes']
        assert num_nodes is not None

        edge_index, edge_weight = get_laplacian(
            edge_index=data['edge_index'],
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



class LapPENodeEncoder(torch.nn.Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, evects, evals, dim_in, dim_pe, dim_emb, model_type, encoder_layers, n_heads, post_layers, 
                 max_freqs, raw_norm_type = "none", pass_as_var=True, expand_x=True):
        super().__init__()
        dim_in = dim_in  # Expected original input node features dim

        dim_pe = dim_pe  # Size of Laplace PE embedding
        model_type = model_type  # Encoder NN model type for PEs
        if model_type not in ['Transformer', 'DeepSet']:
            raise ValueError(f"Unexpected PE model {model_type}")
        self.model_type = model_type
        n_layers = encoder_layers  # Num. layers in PE encoder model
        n_heads = n_heads  # Num. attention heads in Trf PE encoder
        post_n_layers = post_layers  # Num. layers to apply after pooling
        max_freqs = max_freqs  # Num. eigenvectors (frequencies)
        norm_type =   raw_norm_type # Raw PE normalization layer type
        self.pass_as_var = pass_as_var  # Pass PE also as a separate variable
        self.evects = evects
        self.evals = evals
        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"LapPE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = nn.Linear(2, dim_pe)
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        activation = nn.ReLU 
        if model_type == 'Transformer':
            # Transformer model for LapPE
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe,
                                                       nhead=n_heads,
                                                       batch_first=True)
            self.pe_encoder = nn.TransformerEncoder(encoder_layer,
                                                    num_layers=n_layers)
        else:
            # DeepSet model for LapPE
            layers = []
            if n_layers == 1:
                layers.append(activation())
            else:
                self.linear_A = nn.Linear(2, 2 * dim_pe)
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)

        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(activation())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.post_mlp = nn.Sequential(*layers)


    def forward(self, x, device):
       
        EigVals = torch.Tensor(self.evals)
        EigVecs = self.evects
        print("EigVecs shape:", EigVecs.shape)  # Should be (Num nodes, k)
        print("EigVals shape (before unsqueeze):", EigVals.shape)  # Likely (k,)
        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        EigVals = EigVals.unsqueeze(0).repeat(EigVecs.size(0), 1).unsqueeze(2)  # Shape: (Num nodes, k, 1)
        # EigVecs = EigVecs.unsqueeze(2)  # Shape: (Num nodes, k, 1)
        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).to(device) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # PE encoder: a Transformer or DeepSet model
        if self.model_type == 'Transformer':
            pos_enc = self.pe_encoder(src=pos_enc,
                                      src_key_padding_mask=empty_mask[:, :, 0])
        else:
            pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(x)
        else:
            h = x
        # Concatenate final PEs to input embedding
        # x_agg = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            x_pe_LapPE = pos_enc
        return h.detach(), x_pe_LapPE.detach()


def lap_pe_pyg(edge_index, k, num_nodes=None, padding=False, return_eigval=False):
    """PyG version of Laplacian Positional Encoding, following DGL implementation.
    
    Parameters:
    -----------
    edge_index : torch.Tensor
        Edge indices in PyG format (2 x E)
    k : int
        Number of smallest non-trivial eigenvectors to use
    num_nodes : int, optional
        Number of nodes in the graph. If None, inferred from edge_index
    padding : bool, optional
        If False, raise exception when k>=n. Otherwise, add padding
    return_eigval : bool, optional
        If True, return eigenvalues with eigenvectors
        
    Returns:
    --------
    torch.Tensor or (torch.Tensor, torch.Tensor)
        Positional encoding of shape (N, k) or tuple with eigenvalues
    """
    # Get number of nodes if not provided
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    # Check k < n constraint
    if not padding and num_nodes <= k:
        raise ValueError(
            f"Number of eigenvectors k must be smaller than number of nodes n, "
            f"got k={k} and n={num_nodes}"
        )
    
    # Get normalized Laplacian
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)
    )
    
    # Select eigenvectors with smaller eigenvalues
    if k + 1 < num_nodes - 1:
        # Use sparse solver for efficiency when k is small
        EigVal, EigVec = scipy.sparse.linalg.eigs(
            L, k=k + 1, which="SR", ncv=4 * k, tol=1e-2
        )
        # print(EigVal.dtype, EigVec.dtype)
        max_freqs = k
        topk_indices = EigVal.argsort()[1:]  # Skip the first trivial eigenvector
    else:
        # Fallback to dense solver
        EigVal, EigVec = np.linalg.eig(L.toarray())
        max_freqs = min(num_nodes - 1, k)
        kpartition_indices = np.argpartition(EigVal, max_freqs)[: max_freqs + 1]
        topk_eigvals = EigVal[kpartition_indices]
        topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]
    
    # Convert complex values to real
    topk_EigVal = EigVal[topk_indices].real
    topk_EigVec = EigVec[:, topk_indices].real
    
    # Convert to torch tensors
    eigvals = torch.from_numpy(topk_EigVal).float()
    
    # Apply random sign flips to eigenvectors
    rand_sign = 2 * (np.random.rand(max_freqs) > 0.5) - 1.0
    PE = torch.from_numpy(rand_sign * topk_EigVec).float()
    
    # Add padding if needed
    if num_nodes <= k:
        temp_EigVec = torch.zeros(num_nodes, k - num_nodes + 1)
        PE = torch.cat([PE, temp_EigVec], dim=1)
        temp_EigVal = torch.full((k - num_nodes + 1,), float('nan'))
        eigvals = torch.cat([eigvals, temp_EigVal])
    
    if return_eigval:
        return PE, eigvals
    return PE


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs

def laplacian_positional_encoding(dataset, pos_enc_dim):
    """
    Graph positional encoding via Laplacian eigenvectors for PyG.

    Args:
        data: PyG data object
        pos_enc_dim: Number of positional encoding dimensions

    Returns:
        Updated PyG data object with `lap_pos_enc` attribute
    """
    # Convert edge_index to a sparse adjacency matrix
    edge_index = dataset.graph['edge_index']
    edge_weight = None
    n = dataset.graph['num_nodes']
    adj = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=n).astype(float)

    # Compute normalized Laplacian
    degree = sp.diags((adj.sum(axis=1).A1).clip(1) ** -0.5, dtype=float)
    laplacian = sp.eye(n) - degree @ adj @ degree

    # Compute eigenvalues and eigenvectors
    eigval, eigvec = sp.linalg.eigs(laplacian, k=pos_enc_dim + 1, which='SR', tol=1e-2)
    eigvec = eigvec[:, eigval.argsort()]  # Sort eigenvectors by eigenvalues in ascending order

    # Add positional encoding to the node features
    # data.lap_pos_enc = torch.from_numpy(eigvec[:, 1:pos_enc_dim + 1]).float()
    print("Eigen_vec: ", eigvec)
    return eigvec


def edge_index_to_adj_t(edge_index, num_nodes):
    values = torch.ones(edge_index.size(1), device=edge_index.device)  # All edges have weight 1
    adj_t = torch.sparse_coo_tensor(
        edge_index, values, (num_nodes, num_nodes)
    )
    return adj_t

def aggregate_hops_mean(x, edge_index,device):
    ## assumes no duplicates in edge index
    src, dst = edge_index
    src = src.to(device)
    dst = dst.to(device)
    # 1-hop aggregation
    x_1hop = scatter_mean(x[src], dst, dim=0, dim_size=x.size(0))

    # # 2-hop aggregation using adjacency propagation
    x_2hop = scatter_mean(x_1hop[src], dst, dim=0, dim_size=x.size(0))


    return x_1hop, x_2hop

def get_degree(edge_index, num_nodes, node_feat_mean, device):
    """Compute and scale all structural features for undirected graph"""
    node_feat_mean = node_feat_mean.to("cpu")
    # Ensure the graph is undirected 
    # 1. Degree (simpler for undirected graphs)
    deg_start = timer()
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float).to(device)
    deg_end = timer()
    return scale_feature(deg, node_feat_mean)