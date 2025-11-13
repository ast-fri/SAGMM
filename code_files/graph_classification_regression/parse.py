from sagmm import *
def parse_method(args, dataset, device):
    num_tasks = dataset.num_tasks
    if args.method == 'SAGMM':
        print("Loading SAGMM model")
        model = SAGMM(args,num_tasks, args.hidden_channels,args.hidden_channels, graph_weight=args.graph_weight, aggregate=args.aggregate,
                     gnn_num_layers=args.num_layer, gnn_dropout=args.drop_ratio, gnn_use_bn=args.gnn_use_bn, gnn_use_residual=args.gnn_use_residual, gnn_use_weight=args.gnn_use_weight, gnn_use_init=args.gnn_use_init, gnn_use_act=args.gnn_use_act,
                     num_experts=args.num_experts).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def parser_add_main_args(parser):
    ## moe related
    parser.add_argument('--folder_name', type=str, default='temp')
    parser.add_argument('--runs_path', type=str, default='/runs')
    parser.add_argument('--expert_type', type=str, default='multiple_models')
    parser.add_argument('--num_experts', '-n', type=int, default=4,
                        help='total number of experts in GCN-MoE')
    parser.add_argument('--data_dir', type=str, default='dataset/')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')

    # gnn specific
    parser.add_argument('--method', type=str, default='SAGMM')
    parser.add_argument('--use_graph', action='store_true', help='use input graph')
    parser.add_argument('--aggregate', type=str, default='add', help='aggregate type, add or cat.')
    parser.add_argument('--graph_weight', type=float, default=0.8, help='graph weight.')
    parser.add_argument('--gnn_use_bn', action='store_true', help='use batchnorm for each GNN layer')
    parser.add_argument('--gnn_use_ln', action='store_true', help='use layernorm for each GNN layer')
    parser.add_argument('--gnn_use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--gnn_use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--gnn_use_init', action='store_true', help='use initial feat for each GNN layer')
    parser.add_argument('--gnn_use_act', action='store_true', help='use activation for each GNN layer')
    parser.add_argument('--gnn_num_layers', type=int, default=3, help='number of layers for GNN')
    parser.add_argument('--gnn_dropout', type=float, default=0.)
    parser.add_argument('--gnn_weight_decay', type=float, default=1e-3)

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=200, help='early stopping patience.')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to evaluate')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--save_result', action='store_true',
                        help='save result')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--use_pretrained', action='store_true', help='whether to use pretrained model')
    parser.add_argument('--save_att', action='store_true', help='whether to save attention (for visualization)')
    parser.add_argument('--model_dir', type=str, default='../../model/')

    # other gnn parameters (for baselines)
    parser.add_argument('--exclude_expert', type=int, default=0, 
                        help='Number of epochs to train router')
    parser.add_argument('--max_freqs', type=int, default=20, 
                    help='Number of epochs to train router')
    parser.add_argument('--prune_interval', type=int, default=2,
                    help='Interval (in epochs) at which to prune experts.')
    parser.add_argument('--min_experts', type=int, default=1,
                        help='Minimum number of experts to keep in the model.')
    parser.add_argument('--importance_threshold_factor', type=float, default=0.7,
                        help='Factor to determine importance threshold relative to mean importance.')
    parser.add_argument('--gate_method', type=str, default="attention",
                        help='Which gate mechanism to choose')
    # OGBG parameters
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--hidden_channels', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-moltox21",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    ## new hyperparams
    parser.add_argument('--add_imp_loss',action='store_true',
                        help='To decide whether to add importance loss')
    parser.add_argument('--add_diversity_loss',action='store_true',
                        help='To decide whether to add diversity loss')
    parser.add_argument('--add_simple_loss',action='store_true',
                        help='To decide whether to add simple_loss')
    parser.add_argument('--add_l1_loss',action='store_true',
                        help='To decide whether to add simple_loss')
    parser.add_argument('--add_sigmoid',action='store_true',
                        help='To decide whether to add simple_loss')
    parser.add_argument('--add_expert_norm',action='store_true',
                        help='To decide whether to add projection')

    parser.add_argument('--lambda_l1_loss', type=float, default=0.1)
    parser.add_argument('--encode_mode', type=str, default="graph_mean_lap", help="Choose from ['lap', 'multihop_lap']")
    parser.add_argument('--agg_mode', type=str, default="add", help="Choose from ['none', 'concat', 'add']")
    parser.add_argument('--prune_type', type=str, default="gate_scores",
                        help='Which gate mechanism to choose [new_logits, gate_scores]')
    parser.add_argument('--gate_type', type=str, default="zeros",
                        help='Which gate mechanism to choose [zeros, randn]')
    parser.add_argument('--add_proj',action='store_true',
                        help='To decide whether to add projection')
    parser.add_argument('--graph_pooling_type', type=str, default="mean",
                        help='Which pooling  to choose [mean, max, sum, attention, set2set]')
    parser.add_argument('--check_align', action='store_true',
                        help='Whether to check alignment and plot it')

