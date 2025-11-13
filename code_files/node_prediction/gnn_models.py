import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, SAGEConv, GINConv, ChebConv, JumpingKnowledge
from torch_geometric.nn import MixHopConv

class GCNwithJK_P(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True, use_ln=False, use_residual=True, use_act=True, jk_mode='max'):
        super(GCNwithJK_P, self).__init__()
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

        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=True, bias=False))
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

class SGC_P(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True, K=3):
        super(SGC_P, self).__init__()

        self.conv = SGConv(in_channels, hidden_channels, K=K)

        self.dropout = dropout

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

class GCN_P(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(GCN_P, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        # self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, adj_t)
        return x

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(GCN, self).__init__()
        self.use_weight = True
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.W = nn.Linear(hidden_channels, hidden_channels)
        self.bns = nn.ModuleList()
        self.lns = nn.ModuleList() 
        ##option to use layer norm as well as batchnorm
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        if use_ln:
            self.lns.append(nn.LayerNorm(hidden_channels))
        
        for _ in range(num_layers):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=True, bias=False))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))
            
        self.dropout = dropout
        self.activation = F.relu
        self.use_ln = use_ln
        self.use_bn = use_bn
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

    def forward(self, x, edge_index):
        x = self.fcs[0](x)
        if self.use_ln:
            x = self.lns[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_weight:
                x = self.W(x)
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_ln:  
                x = self.lns[i+1](x)  
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.out_lin(x)
        return x
    
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(GAT, self).__init__()
        self.use_weight = True
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.W = nn.Linear(hidden_channels, hidden_channels)
        
        self.bns = nn.ModuleList()  # BatchNorms
        self.lns = nn.ModuleList()  # LayerNorms
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))  # First BatchNorm
        if use_ln:
            self.lns.append(nn.LayerNorm(hidden_channels))  # First LayerNorm

        for _ in range(num_layers):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, bias=False))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))

        # self.out_lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout
        self.activation = F.relu
        self.use_ln = use_ln
        self.use_bn = use_bn
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
        self.out_lin.reset_parameters()

    def forward(self, x, edge_index):
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        if self.use_ln:
            x = self.lns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

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
        # x = self.out_lin(x)
        return x
    
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(GraphSAGE, self).__init__()
        self.use_weight = True
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.W = nn.Linear(hidden_channels, hidden_channels)

        self.bns = nn.ModuleList()  # BatchNorms
        self.lns = nn.ModuleList()  # LayerNorms
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        if use_ln:
            self.lns.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, bias=False))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))

        # self.out_lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout
        self.activation = F.relu
        self.use_ln = use_ln
        self.use_bn = use_bn
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
        self.out_lin.reset_parameters()

    def forward(self, x, edge_index):
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        if self.use_ln:
            x = self.lns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

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

        return x

class SGC(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(SGC, self).__init__()
        self.use_weight = True
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.W = nn.Linear(hidden_channels, hidden_channels)

        self.bns = nn.ModuleList()  # BatchNorms
        self.lns = nn.ModuleList()  # LayerNorms
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        if use_ln:
            self.lns.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers):
            self.convs.append(
                SGConv(hidden_channels, hidden_channels, bias=False))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))

        # self.out_lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout
        self.activation = F.relu
        self.use_ln = use_ln
        self.use_bn = use_bn
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
        self.out_lin.reset_parameters()

    def forward(self, x, edge_index):
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        if self.use_ln:
            x = self.lns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

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
        # x = self.out_lin(x)
        return x

class GCNwithJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True, use_ln=False, use_residual=True, use_act=True, jk_mode='max'):
        super(GCNwithJK, self).__init__()
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
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=True, bias=False))
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

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True, use_ln=False, use_residual=True, use_act=True):
        super(GIN, self).__init__()
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
            nn_layers = nn.Sequential(nn.Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(nn_layers, train_eps=True))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))

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

        return x

class ChebNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 K=2, dropout=0.5, save_mem=True, use_bn=True, use_ln=False, use_residual=True, use_act=True):
        super(ChebNet, self).__init__()
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
            self.convs.append(ChebConv(hidden_channels, hidden_channels, K=K, normalization="sym", bias=False))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))

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

        return x

class MixHopNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, powers=[1, 2, 3], use_bn=True, use_ln=False, use_act=True):
        super(MixHopNet, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers):
            self.convs.append(MixHopConv(hidden_channels, hidden_channels, powers=powers, bias=False))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels * len(powers)))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels * len(powers)))

        self.W = nn.Linear(hidden_channels * len(powers), hidden_channels)

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.use_act = use_act
        self.use_weight = True
    
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
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            if self.use_ln:
                x = self.lns[i](x)
            if self.use_weight:
                x = self.W(x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x
