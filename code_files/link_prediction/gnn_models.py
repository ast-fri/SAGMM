import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, SGConv, JumpingKnowledge



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, normalize=False, cached=False):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=normalize, cached=cached))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=normalize, cached=cached))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=normalize, cached=cached))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True, K=3):
        super(SGC, self).__init__()

        self.conv = SGConv(in_channels, out_channels, K=K)

        self.dropout = dropout

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x


class GCNwithJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True, use_ln=False, use_residual=True, use_act=True, jk_mode='max',normalize=False, cached=False):
        super(GCNwithJK, self).__init__()
    
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        

        self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached, normalize=normalize))

        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        if use_ln:
            self.lns.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=normalize))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached, normalize=normalize))
        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=1)
        self.fc_out = nn.Linear(num_layers * hidden_channels if jk_mode == 'cat' else hidden_channels, out_channels)

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.use_residual = use_residual
        self.use_act = use_act
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_ln:
            for ln in self.lns:
                ln.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()
        self.jk.reset_parameters()
        self.fc_out.reset_parameters()
        # self.out_lin.reset_parameters()

    def forward(self, x, edge_index):
        
        layer_outs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            if self.use_ln:
                x = self.lns[i](x)
            x = self.activation(x)
            layer_outs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.convs[-1](x, edge_index)
        layer_outs.append(x)
        x = self.jk(layer_outs)
        x = self.fc_out(x)
        return x

class GCNwithJK_O(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=None, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True, use_ln=False, use_residual=True, use_act=True, jk_mode='max', normalize=False, cached=False):
        super(GCNwithJK_O, self).__init__()
        self.use_weight = True
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lns = nn.ModuleList()

        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.W = nn.Linear(hidden_channels, hidden_channels)

        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        if use_ln:
            self.lns.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=normalize, bias=False))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))

        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=num_layers)
        self.fc_out = nn.Linear(num_layers * hidden_channels if jk_mode == 'cat' else hidden_channels, hidden_channels)

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.use_residual = use_residual
        self.use_act = use_act
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_ln:
            for ln in self.lns:
                ln.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        self.W.reset_parameters()
        # self.out_lin.reset_parameters()

    def forward(self, x, edge_index):
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        if self.use_ln:
            x = self.lns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_outs = [x]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_weight:
                x = self.W(x)
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_ln:
                x = self.lns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outs.append(x)

        x = self.jk(layer_outs)
        return self.fc_out(x)