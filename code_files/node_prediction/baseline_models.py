import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, SAGEConv, GINConv, LEConv, ChebConv, JumpingKnowledge, APPNP, MessagePassing 
from torch_geometric.nn import MixHopConv



class indi_SGC_P(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True, K=3):
        super(indi_SGC_P, self).__init__()

        self.conv = SGConv(in_channels, hidden_channels, K=K)
        self.out_lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.out_lin.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.out_lin(x)
        return x


class indi_GCN_P(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(indi_GCN_P, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, num_classes))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class indi_MLP_P(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2,
                 dropout=0.5, use_bn=True, use_ln=False, use_residual=False, use_act=True):
        super(indi_MLP_P, self).__init__()

        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        

        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        self.dropout = dropout

        # First layer
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        # self.res_linears.append(nn.Linear(in_channels, hidden_channels) if use_residual else nn.Identity())

        # Hidden layers (no output layer!)
        for _ in range(num_layers - 1):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, num_classes))   

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        

    def forward(self, x, edge_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            h_res = x
            x = lin(x)
            if self.use_bn:
                x = self.bns[i](x)
            if self.use_act:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x) 
        return x  


class indi_SAGE_P(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 dropout=0.5, use_bn=True, use_ln=False, use_residual=True, use_act=True):
        super(indi_SAGE_P, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.res_linears = nn.ModuleList()
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        self.dropout = dropout
        self.num_layers = num_layers

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, bias=False))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.res_linears.append(nn.Linear(in_channels, hidden_channels) if use_residual else nn.Identity())

        # Hidden layers
        for _ in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, bias=False))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.res_linears.append(nn.Identity() if use_residual else nn.Identity())
        
        # Last layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, bias=False))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.res_linears.append(nn.Identity() if use_residual else nn.Identity())
        self.mlp = indi_MLP_P(in_channels + hidden_channels * num_layers, 2 * num_classes, num_classes, num_layers=2, use_bn=True)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for res in self.res_linears:
            if hasattr(res, 'reset_parameters'):
                res.reset_parameters()

    def forward(self, x, edge_index):

        collect = []

        collect.append(x)
        for i in range(self.num_layers):
            h_res = x  # Residual
            x = self.convs[i](x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            if self.use_act:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            collect.append(x)
            if self.use_residual:
                x = x + self.res_linears[i](h_res)
        return self.mlp(torch.cat(collect, -1))

class indi_GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(indi_GraphSAGE, self).__init__()
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

        self.out_lin = nn.Linear(hidden_channels, num_classes)
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
        x = self.out_lin(x)
        return x


class indi_GCN_PR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(indi_GCN_PR, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, adj_t)
        return x


class indi_SAGE_PR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(indi_SAGE_PR, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        # self.convs.append(SAGEConv(hidden_channels, out_channels))

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

class indi_MLP_PR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(indi_MLP_PR, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        # self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, edge_index=None):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.lins[-1](x)
        return x

class indi_FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(indi_FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x

class indi_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(indi_GCN, self).__init__()
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
        self.out_lin = nn.Linear(hidden_channels, num_classes)
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
        x = self.out_lin(x)
        return x
    
class indi_GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes,num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(indi_GAT, self).__init__()
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

        self.out_lin = nn.Linear(hidden_channels, num_classes)
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
        x = self.out_lin(x)
        return x
    

class indi_SGC(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes,num_layers=2,
                 dropout=0.5, save_mem=True, use_ln=False, use_bn=True, use_residual=True, use_act=True):
        super(indi_SGC, self).__init__()
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

        self.out_lin = nn.Linear(hidden_channels, num_classes)
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
        x = self.out_lin(x)
        return x

class indi_GCNwithJK(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True, use_ln=False, use_residual=True, use_act=True, jk_mode='max'):
        super(indi_GCNwithJK, self).__init__()
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
        self.out_lin = nn.Linear(hidden_channels, num_classes)
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
        self.out_lin.reset_parameters()

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
        x = self.fc_out(x)
        x = self.out_lin(x)
        return x

class indi_GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True, use_ln=False, use_residual=True, use_act=True):
        super(indi_GIN, self).__init__()
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

class indi_ChebNet(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 K=3, dropout=0.5, save_mem=True, use_bn=True, use_ln=False, use_residual=True, use_act=True):
        super(indi_ChebNet, self).__init__()
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

class indi_MixHopNet(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_classes, num_layers=2,
                 dropout=0.5, powers=[1, 2, 3], use_bn=True, use_ln=False, use_act=True):
        super(indi_MixHopNet, self).__init__()

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
