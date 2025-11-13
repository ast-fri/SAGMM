from sagmm import SAGMM
from baseline_models import indi_GCN_P, indi_SAGE_P, indi_MLP_P, indi_GraphSAGE, indi_SGC_P, indi_GCN, indi_GAT, indi_SGC, indi_GCNwithJK
def parse_method(args, c, d, n, device):
    if args.method == 'SAGMM':
        model = SAGMM(args, n, d, args.hidden_channels, c, gnn_num_layers=args.gnn_num_layers, gnn_dropout=args.gnn_dropout,
                     num_experts=args.num_experts,k=args.k,coef=args.coef, device=device).to(device)
                
    elif args.method == 'mlp_p':
        model = indi_MLP_P(in_channels=d, hidden_channels=args.hidden_channels, num_layers=args.gnn_num_layers, num_classes = c, dropout=args.gnn_dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln).to(device)
    elif args.method == 'gcn_p':
        model = indi_GCN_P(in_channels=d, hidden_channels=args.hidden_channels, num_layers=args.gnn_num_layers, num_classes = c, dropout=args.gnn_dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln).to(device)
    elif args.method == 'sage_p' or args.method == 'sage':
        model = indi_GraphSAGE(in_channels=d, hidden_channels=args.hidden_channels, num_layers=args.gnn_num_layers, num_classes = c, dropout=args.gnn_dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln).to(device)
    elif args.method == 'l_sage_p':
        model = indi_SAGE_P(in_channels=d, hidden_channels=args.hidden_channels, num_layers=args.gnn_num_layers, num_classes = c, dropout=args.gnn_dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln).to(device)
    elif args.method == 'sgc_p':
        model = indi_SGC_P(in_channels=d, hidden_channels=args.hidden_channels, num_layers=args.gnn_num_layers, num_classes = c, dropout=args.gnn_dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln, K=3).to(device)
    elif args.method == 'gcn':
        model = indi_GCN(in_channels=d, hidden_channels=args.hidden_channels, num_layers=args.gnn_num_layers, num_classes = c, dropout=args.gnn_dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln).to(device)
    elif args.method == 'sgc':
        model = indi_SGC(in_channels=d, hidden_channels=args.hidden_channels, num_layers=args.gnn_num_layers, num_classes = c, dropout=args.gnn_dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln).to(device)
    elif args.method == 'gat':
        model = indi_GAT(in_channels=d, hidden_channels=args.hidden_channels, num_layers=args.gnn_num_layers, num_classes = c, dropout=args.gnn_dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln).to(device)
    elif args.method == 'gcnwithjk_p':
        model = indi_GCNwithJK(in_channels=d, hidden_channels=args.hidden_channels, num_layers=args.gnn_num_layers, num_classes = c, dropout=args.gnn_dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln).to(device)
    
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    
    # Dataset and evaluation
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv') # 'ogbn-proteins' 'ogbn-arxiv'
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str) 
    parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10, help='number of distinct runs')
    parser.add_argument('--directed', action='store_true', help='set to not symmetrize adjacency')
    parser.add_argument('--folder_name', type=str, default='temp')
    parser.add_argument('--runs_path', type=str)
    parser.add_argument('--protocol', type=str, default='semi', help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true', help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes randomly selected')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'], help='evaluation metric')
    parser.add_argument('--method', type=str, default='SAGMM')
    parser.add_argument('--aggregate', type=str, default='add', help='aggregate type, add or cat.')
    parser.add_argument('--pin_memory',action='store_true', help='Flag to pin the tensor')
    parser.add_argument('--train_prop', type=float, default=.5, help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25, help='validation label proportion')
    # gnn parameters
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--gnn_use_bn', action='store_true', help='use batchnorm for each GNN layer')
    parser.add_argument('--gnn_use_ln', action='store_true', help='use layernorm for each GNN layer')
    parser.add_argument('--gnn_use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--gnn_use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--gnn_use_init', action='store_true', help='use initial feat for each GNN layer')
    parser.add_argument('--gnn_use_act', action='store_true', help='use activation for each GNN layer')
    parser.add_argument('--gnn_num_layers', type=int, default=3, help='number of layers for GNN')
    parser.add_argument('--gnn_dropout', type=float, default=0.)
    parser.add_argument('--gnn_weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=16934, help='mini batch training for large graphs')
    parser.add_argument('--display_step', type=int, default=1, help='how often to print')
    parser.add_argument('--eval_step', type=int, default=1, help='how often to evaluate')
    parser.add_argument("--fan_out", type=str, default="15,10,5", help="Fan-out for each GNN layer")
    
    # utility
    parser.add_argument('--cached', action='store_true', help='set to use faster sgc')
    parser.add_argument('--save_result', action='store_true', help='save result')

    # pretraining parameters
    parser.add_argument('--use_pretrained', action='store_true', help='whether to use pretrained model')
    parser.add_argument('--expert_data_percent', type=float, default=0.5, help='Percentage of training data for expert training (between 0 and 1)')
    parser.add_argument('--router_data_percent', type=float, default=0.5, help='Percentage of training data for router training (between 0 and 1)')
    parser.add_argument('--last_layer',action='store_true', help='To decide whether to keep last layer of each model (would be needed when pretraining the model)')
    
    # Gating parameters
    parser.add_argument('--num_experts', '-n', type=int, default=2, help='total number of experts')
    parser.add_argument('--max_experts', type=int, default=2, help='Maximum number of experts in the model.')
    parser.add_argument('--min_experts', type=int, default=1,help='Minimum number of experts to keep in the model.')
    parser.add_argument('--importance_threshold_factor', type=float, default=0.4,help='Factor to determine importance threshold relative to mean importance.')
    parser.add_argument('--prune_interval', type=int, default=50,help='Interval (in epochs) at which to prune experts.')
    parser.add_argument('--gate_method', type=str, default="attention", help='Which gate mechanism to choose inside SAGMM [attention or top_any]')
    parser.add_argument('--router_method', type=str, default="attention",help='Which router mechanism to choose [SAGMM or noisy top-k gating]')
    parser.add_argument('--individual_model', action='store_true', help='Flag to train individual model')
    parser.add_argument('--add_proj',action='store_true', help='To decide whether to add projection')
    parser.add_argument('--add_sigmoid',action='store_true', help='To decide whether to pass scores through sigmoid')
    parser.add_argument('--add_expert_norm',action='store_true', help='To decide whether to normalize the expert outputs')
    parser.add_argument('--add_exp_l_norm', action='store_true',help='Include layer norm as expert norm')
    parser.add_argument('--max_freqs', type=int, default=128,  help='Tells extra dimension added when adding grap information')
    parser.add_argument('--encode_mode', type=str, default="none", help="Choose from ['multihop_lap', 'lap', 'RWSE', 'n2v']")
    parser.add_argument('--agg_mode', type=str, default="none", help="Choose from ['none', 'concat', 'add']")
    parser.add_argument('--prune_type', type=str, default="gate_scores", help='Which gate mechanism to choose [new_logits, gate_scores]')
    parser.add_argument('--gate_type', type=str, default="randn", help='Which gate mechanism to choose [zeros, randn]')
    parser.add_argument('--stop_prune',action='store_true',help='To turn off pruning')
    parser.add_argument('--div_weight', type=float, default=0.01)
    parser.add_argument('--imp_weight', type=float, default=1)
    parser.add_argument('--k', type=int, default=2, help='selected number of experts')
    parser.add_argument('--coef', type=float, default=1, help='loss coefficient for load balancing loss in sparse MoE training')   
    
    # Loss terms
    parser.add_argument('--add_imp_loss',action='store_true', help='To decide whether to add importance loss')
    parser.add_argument('--add_diversity_loss',action='store_true', help='To decide whether to add diversity loss')
   
    
    
  
    
  



