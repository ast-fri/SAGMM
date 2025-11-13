import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
import tqdm
@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        if args.expert_type == "multiple_models":
            out, _ , _= model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        else:
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    valid_loss = 0
    return train_acc, valid_acc, test_acc, valid_loss, out

@torch.no_grad()
def evaluate_expert(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out= model(dataset.graph['node_feat'], dataset.graph['edge_index'])
    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    valid_loss = 0
    return train_acc, valid_acc, test_acc, valid_loss, out


@torch.no_grad()
def evaluate_large(model, dataset, split_idx, eval_func, criterion, args, device="cpu", result=None):
    # print("print device: ",device)
    if result is not None:
        out = result
    else:
        model.eval()
    device = torch.device(device)
    model.to(device)
    dataset.label = dataset.label.to(device)   
    edge_index = dataset.graph['edge_index']  
    x = dataset.graph['node_feat']  
    # Move split_idx to the same device
    split_idx = {k: v.to(device) for k, v in split_idx.items()}
    if args.expert_type == "multiple_models":
        out, _, _ = model(x, edge_index)
    else:
        out = model(x, edge_index)
        # print(out)
    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    return train_acc, valid_acc, test_acc, 0, out

@torch.no_grad()
def evaluate_large_gpu(model, dataset, split_idx, eval_func, criterion, args, device="cuda", result=None, batch_size=1024):
    """
    Evaluate model in batches on GPU, to avoid OOM while loading the entire dataset.
    """
    model.eval()

    n = dataset.graph['num_nodes']
    edge_index = dataset.graph['edge_index']
    x = dataset.graph['node_feat']
    label = dataset.label

    # Ensure data is on CPU to begin with
    x = x.to('cpu')
    edge_index = edge_index.to('cpu')
    label = label.to('cpu')

    # Ensure split indices are on CPU
    split_idx_cpu = {k: v.to('cpu') for k, v in split_idx.items()}

    # Evaluation results
    eval_results = {}
    splits = ['train', 'valid', 'test']
    for split in splits:
        idx_split = split_idx_cpu[split]
        y_true = label[idx_split]  # Now both label and idx_split are on CPU
        y_pred = []

        batch_size = args.batch_size  # Define evaluation batch size

        num_batches = len(idx_split) // batch_size + int(len(idx_split) % batch_size > 0)
        for i in range(num_batches):
            idx_batch = idx_split[i * batch_size : (i + 1) * batch_size]
            # idx_batch is already on CPU

            # Get subgraph induced by idx_batch
            # Since subgraph with relabel_nodes=True, nodes are relabeled from 0
            # We need to build a mapping from original node indices to subgraph indices
            subset, edge_index_batch, edge_attr_batch = subgraph(
                idx_batch, edge_index, relabel_nodes=True, num_nodes=n, return_edge_mask=True)
            
            # subset contains the nodes in the subgraph (original indices)
            # Build mapping: original node index -> subgraph node index
            mapping = {orig_idx.item(): i for i, orig_idx in enumerate(subset)}
            idx_batch_subgraph = torch.tensor([mapping[idx.item()] for idx in idx_batch], dtype=torch.long)
            # Move data to GPU
            x_sub = x[subset].to(device)
            edge_index_batch = edge_index_batch.to(device)
            model = model.to(device)
            # Forward pass
            with torch.no_grad():
                out_batch, _, _ = model(x_sub, edge_index_batch)

            # Get the output for nodes in idx_batch
            out_batch = out_batch[idx_batch_subgraph]
            y_pred.append(out_batch.cpu())
            # Clean up to free GPU memory
            del x_sub, edge_index_batch, out_batch
            torch.cuda.empty_cache()

        # Concatenate outputs
        y_pred = torch.cat(y_pred, dim=0)

        # Compute evaluation metric
        acc = eval_func(y_true, y_pred)
        eval_results[split] = acc

    train_acc = eval_results['train']
    valid_acc = eval_results['valid']
    test_acc = eval_results['test']


    return train_acc, valid_acc, test_acc, 0, None

@torch.no_grad()
def evaluate_large_expert(model, dataset, split_idx, eval_func, criterion, args, device="cpu", result=None):
    if result is not None:
        out = result
    else:
        model.eval()
    device = torch.device(device)
    model.to(device)
    dataset.label = dataset.label.to(device)  
    edge_index = dataset.graph['edge_index']  
    x = dataset.graph['node_feat']  
    # Move split_idx to the same device
    split_idx = {k: v.to(device) for k, v in split_idx.items()}   
    out = model(x, edge_index)
    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    return train_acc, valid_acc, test_acc, 0, out

@torch.no_grad()
def evaluate_batch(model, dataset, split_idx, eval_func, args, device, n, true_label):
    num_batch = n // args.batch_size + int(n % args.batch_size > 0)
    
    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    valid_mask = torch.zeros(n, dtype=torch.bool)
    valid_mask[split_idx['valid']] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[split_idx['test']] = True
    model.to(device)
    model.eval()
    # Process nodes in order instead of random permutation
    idx = torch.arange(n)
    # Lists to collect node indices and outputs
    all_idx = []
    all_out = []
    with torch.no_grad():
        for i in range(num_batch):
            idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
            x_i = x[idx_i].to(device)

            # Extract subgraph induced by idx_i
            edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
            edge_index_i = edge_index_i.to(device)
            # Forward pass
            out_i, _, _ = model(x_i, edge_index_i)
            # Collect the outputs and corresponding node indices
            all_idx.append(idx_i)
            all_out.append(out_i)  # Move outputs to CPU if necessary
    # After processing all batches, concatenate the indices and outputs
    all_idx = torch.cat(all_idx)
    all_out = torch.cat(all_out)
    # Sort the outputs according to node indices to ensure alignment
    sorted_idx, order = torch.sort(all_idx)
    all_out = all_out[order]
    # Now, all_out is aligned with node indices from 0 to n-1
    # Get the true labels and ensure they are on CPU
    y_true = true_label

    # Use the masks to select predictions and labels for each split
    train_pred = all_out[train_mask]
    train_true = y_true[train_mask]
    valid_pred = all_out[valid_mask]
    valid_true = y_true[valid_mask]
    test_pred = all_out[test_mask]
    test_true = y_true[test_mask]
    # Compute evaluation metrics using eval_func
    train_acc = eval_func(train_true, train_pred)
    valid_acc = eval_func(valid_true, valid_pred)
    test_acc = eval_func(test_true, test_pred)

    return train_acc, valid_acc, test_acc, 0, None
def eval_acc(true, pred):
    '''
    true: (n, 1)
    pred: (n, c)
    '''
    pred=torch.max(pred,dim=1,keepdim=True)[1]
    # cmp=torch.eq(true, pred)
    # print(f'pred:{pred}')
    # print(cmp)
    true_cnt=(true==pred).sum()

    return true.shape[0], true_cnt.item()
if __name__=='__main__':
    x=torch.arange(4).unsqueeze(1)
    y=torch.Tensor([[3,0,0,0],
                    [3,2,1.5,2.8],
                    [0,0,2,1],
                    [0,0,1,3]
                    ])
    a, b=eval_acc(x, y)
    print(x)
    print(a,b)
