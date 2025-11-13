import torch

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, args, x_agg=None):
    model.eval()
    if(args.individual_model):
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
    else:
        out, _ , _ = model(dataset.graph['node_feat'], dataset.graph['edge_index'], x_agg=x_agg)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    valid_loss = 0
    # print("Train acc: ", train_acc)
    return train_acc, valid_acc, test_acc, valid_loss, 0

@torch.no_grad()
def evaluate_large(model, dataset, split_idx, eval_func, args, x_agg=None,  device="cpu", result=None):
    model.eval()

    # model.eval()
    device = torch.device(device)
    model.to(device)
    dataset.label = dataset.label.to(device)
    
    edge_index = dataset.graph['edge_index']
    
    x = dataset.graph['node_feat']
    
    
    # Move split_idx to the same device
    split_idx = {k: v.to(device) for k, v in split_idx.items()}
    if(args.method != "SAGMM"):
        out = model(x, edge_index)
    else:
        out, _, _ = model(x, edge_index, x_agg=x_agg)
    
    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])


    return train_acc, valid_acc, test_acc, 0, 0


