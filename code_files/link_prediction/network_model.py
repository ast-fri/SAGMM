
import torch
from sagmm_gating import SAGMM_network


class NetworkModel(torch.nn.Module):
    def __init__(self, args, n, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_experts=4, k=1, coef=1e-2, device=None):
        super(NetworkModel, self).__init__()
        self.num_layers = num_layers
        self.experts = torch.nn.ModuleList()
        networks = SAGMM_network(args, n, input_size=in_channels, hid_size=hidden_channels, num_classes=out_channels, num_layers=num_layers, dropout=dropout, num_experts=num_experts, k=k, coef=coef)
        
        self.experts.append(networks)

    def reset_parameters(self):
        for conv in self.experts:
            conv.reset_parameters()

    def forward(self, x, edge_index, x_agg=None, adj_t_norm = None, last_idx=False):
    
        for expert in self.experts:
            
            x, _layer_load_balance_loss, raw_logits = expert(x, edge_index, x_agg=x_agg, adj_t_norm=adj_t_norm, last_idx=last_idx)
        
        return x, _layer_load_balance_loss, raw_logits
