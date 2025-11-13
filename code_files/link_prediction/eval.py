import torch
import torch.nn.functional as F

from torch_geometric.utils import subgraph
import tqdm
from torch.utils.data import DataLoader
from timeit import default_timer as timer

@torch.no_grad()
def evaluate_link(model, predictor, data, split_edge, evaluator, batch_size, method, device, ep, x_agg, K, dataset):
    model.eval()
    predictor.eval()
  
    if(method == 'SAGMM'):
        if(dataset == "ogbl-ppa"):
            h, _, _ = model(data.x, data.adj_t, x_agg=x_agg, adj_t_norm=data.adj_t_norm)
        else:
            h, _, _ = model(data.x, data.adj_t, x_agg=x_agg)
    else:
        if(dataset == "ogbl-ppa" and (method == "gcnwithjk" or method == "gcn")):
            h = model(data.x, data.adj_t_norm)
        else:
            h = model(data.x, data.adj_t)
   
    link_s = timer()
    if(dataset == "ddi"):
        pos_train_edge = split_edge['eval_train']['edge'].to(h.device)
    else:
        pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    # for K in [100]:
    evaluator.K = K
    train_hits = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_valid_pred,
    })[f'hits@{K}']
    valid_hits = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })[f'hits@{K}']
    test_hits = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })[f'hits@{K}']
  

    results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results
