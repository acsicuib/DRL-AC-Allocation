import torch

def aggr_obs(obs_mb, n_node):
    idxs = obs_mb.coalesce().indices()
    vals = obs_mb.coalesce().values()
    new_idx_row = idxs[1] + idxs[0] * n_node
    new_idx_col = idxs[2] + idxs[0] * n_node
    idx_mb = torch.stack((new_idx_row, new_idx_col))

    adj_batch = torch.sparse.FloatTensor(indices=idx_mb,
                                         values=vals,
                                         size=torch.Size([obs_mb.shape[0] * n_node,
                                                          obs_mb.shape[0] * n_node]),
                                         ).to(obs_mb.device)
    return adj_batch


def dag_pool(graph_pool_type, batch_size, n_nodes, device):
    
    if graph_pool_type == 'average':
        elem = torch.full(size=(batch_size[0] * n_nodes, 1),  fill_value=1 / n_nodes, dtype=torch.float32, device=device).view(-1)
    else:
        elem = torch.full(size=(batch_size[0] * n_nodes, 1), fill_value=1, dtype=torch.float32, device=device).view(-1)
    
    idx_0 = torch.arange(start=0, end=batch_size[0], device=device,  dtype=torch.long)
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size[0]*n_nodes, 1)).squeeze()
    idx_1 = torch.arange(start=0, end=n_nodes*batch_size[0], device=device, dtype=torch.long)
    idx = torch.stack((idx_0, idx_1))
    graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([batch_size[0], n_nodes*batch_size[0]])).to(device)

    return graph_pool


