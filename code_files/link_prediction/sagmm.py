
import torch.nn as nn
from network_model import NetworkModel

class SAGMM(nn.Module):
    def __init__(self, args, n, in_channels, hidden_channels, out_channels,
                 gnn_num_layers=1, gnn_dropout=0.5, aggregate='add',num_experts=8,k=4,coef=1, device=None):
        super().__init__()
        self.network_model = NetworkModel(args, n, in_channels, hidden_channels, out_channels,gnn_num_layers, gnn_dropout,num_experts=num_experts, k=k, coef=coef, device=device)
    
        if aggregate == 'add':
            self.fc = nn.Linear(1* out_channels, out_channels)
            
        elif aggregate == 'cat':
            self.fc = nn.Linear(1 * 1 * out_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

    def forward(self, x, edge_index, x_agg=None, adj_t_norm=None, last_idx=False):
        x2, loss, raw_logits = self.network_model(x, edge_index, x_agg=x_agg, adj_t_norm=adj_t_norm, last_idx=last_idx)
        x=self.fc(x2)
        return x, loss, raw_logits

    def reset_parameters(self):
        self.network_model.reset_parameters()