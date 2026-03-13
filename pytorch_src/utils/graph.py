import torch
from typing import List, Tuple


def edge_to_adj_matrix(
    edge_index: torch.LongTensor, num_nodes: int
) -> torch.BoolTensor:
    """
    Convert edge index to adjacency matrix.

    Parameters
    ----------
    edge_index: torch.LongTensor
        edge index, shape [2, E]
    num_nodes: int
        number of nodes

    Returns
    -------
    adj: torch.BoolTensor
        adjacency matrix, shape [N, N], where N is num_nodes
    """
    device = edge_index.device
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)

    if edge_index.numel() == 0:
        return adj

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E].")

    if not (0 <= edge_index).all() or not (edge_index < num_nodes).all():
        raise ValueError("edge_index contains out-of-range node indices.")

    adj[edge_index[0], edge_index[1]] = True

    return adj


def reindex_edge_index(
    edge_index: torch.LongTensor, node_ids: torch.LongTensor
) -> torch.LongTensor:
    """
    Reindex edge_index from node IDs to node indices. It is to ensure the edge_index is compatible with the node features tensor. For example, if node_ids = [10, 20, 30], and edge_index contains edges (10→20) and (20→30), then the reindexed edge_index will contain edges (0→1) and (1→2).

    Parameters
    ----------
    edge_index: torch.LongTensor
        edge index with node IDs, shape [2, E]
    node_ids: torch.LongTensor
        node IDs corresponding to node indices, shape [N]

    Returns
    -------
    edge_index_reindexed: torch.LongTensor
        edge index with node indices, shape [2, E]

    Examples
    --------
    >>> node_ids = torch.tensor([10, 20, 30])
    >>> edge_index = torch.tensor([[10, 20], [20, 30]])
    >>> edge_index_reindexed = reindex_edge_index(edge_index, node_ids)
    >>> print(edge_index_reindexed)
    >>> torch.tensor([[0, 1], [1, 2]])
    """
    max_id = node_ids.max()
    mapping = torch.zeros(max_id + 1, dtype=torch.long, device=edge_index.device)
    mapping[node_ids] = torch.arange(len(node_ids), device=edge_index.device)
    return mapping[edge_index]


@torch.jit.script
def pack_to_graph_batches(
    x: torch.Tensor, t: List[torch.Tensor], batch: torch.LongTensor
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.BoolTensor]:
    """
    Pack node features into graph-batched format, padding with zeros for graphs with fewer nodes. Empty graphs should never occur.

    Parameters
    ----------
    x: torch.Tensor
        Node features, shape [N_total, D], where N_total is total number of nodes across all graphs, D is feature dimension.
    t: List[torch.Tensor]
        List of additional node-level tensors to pack, each of shape [N_total, D_t], where D_t is the feature dimension of that tensor.
    batch: torch.LongTensor
        Batch vector, shape [N_total], batch[i] = graph index of node i.

    Returns
    -------
    outputs: tuple
        A tuple containing:
    x_graphs: List[torch.Tensor]
        List of node feature tensors in graph-batched format. The first element is the packed x tensor of shape [B, L_max, D], where B is batch size, L_max is max number of nodes per graph, D is feature dimension. Subsequent elements are the packed tensors from t, each of shape [B, L_max, D_t].
    idx_out: List[torch.Tensor]
        List of (global) index tensors for each graph in the batch.
    mask_out: torch.BoolTensor
        Mask tensor indicating valid nodes in the graph-batched format, shape [B, L_max].

    Examples
    --------
    >>> x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])  # [6, 1]
    >>> batch = torch.tensor([0, 0, 1, 1, 0, 1])  # graph 0: nodes 0,1,4; graph 1: nodes 2,3,5
    >>> x_graphs, idx_out, mask_out = pack_to_graph_batches(x, [], batch=batch)
    >>> x_graph = x_graphs[0]
    >>> print(x_graph)
    >>> torch.tensor([[[1.0], [2.0], [5.0]],
                      [[3.0], [4.0], [6.0]]])
    >>> print(idx_out)
    >>> [torch.tensor([0, 1, 4]), torch.tensor([2, 3, 5])]
    """
    device = x.device
    B = int(batch.max().item()) + 1  # number of graphs in the batch

    batch_unique = torch.unique(batch)
    if (
        batch_unique.numel() != B
        or batch_unique.min() != 0
        or batch_unique.max() != B - 1
    ):
        raise RuntimeError("Batch IDs must be contiguous starting at 0.")

    N, D = x.size()  # total number of nodes, feature dimension
    L_max = int(
        batch.bincount(minlength=B).max().item()
    )  # max number of nodes per graph

    for t_ in t:
        assert t_.size(0) == N, "All input tensors must have same first dimension as x."
        assert t_.dim() == 2, "All input tensors must be 2D, same as x."

    outs = []
    outs.append(torch.zeros((B, L_max, D), device=device, dtype=x.dtype))
    for t_ in t:
        _, D_t = t_.size()
        outs.append(torch.zeros((B, L_max, D_t), device=device, dtype=t_.dtype))

    mask_out = torch.zeros((B, L_max), dtype=torch.bool, device=device)
    idx_out: List[torch.Tensor] = []

    global_idx = torch.arange(batch.size(0), device=x.device)
    for b in torch.unique(batch, sorted=True):
        mask = batch == b
        L_b = mask.sum().item()  # number of nodes in graph b
        idx = global_idx[mask]  # shape [L_b]

        outs[0][b, :L_b, :] = x[mask]
        for i, t_ in enumerate(t, start=1):
            outs[i][b, :L_b, :] = t_[mask]

        mask_out[b, :L_b] = True
        idx_out.append(idx)

    return outs, idx_out, mask_out


@torch.jit.script
def reorder_from_graph_batches(
    x_graph: torch.Tensor, idx_out: List[torch.Tensor]
) -> torch.Tensor:
    """
    Reorder node features from graph-batched format back to original node order.

    Parameters
    ----------
    x_graph: torch.Tensor
        Node features in graph-batched format, shape [B, L_max, D], where B is batch size, L_max is max number of nodes per graph, D is feature dimension.
    idx_out: List[torch.Tensor]
        List of (global) index tensors for each graph in the batch.

    Returns
    -------
    x: torch.Tensor
        Node features reordered to original node order, shape [N_total, D], where N_total is total number of nodes across all graphs. Note that graphs with fewer than L_max nodes are truncated to their actual lengths.

    Examples
    --------
    >>> x_graph = torch.tensor([[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]])  # [2, 2, 2]
    >>> idx_out = [torch.tensor([0, 2]), torch.tensor([1, 3])]
    >>> x_reordered = reorder_from_graph_batches(x_graph, idx_out)
    >>> print(x_reordered)
    >>> torch.tensor([[1.0, 10.0],
                      [3.0, 30.0],
                      [2.0, 20.0],
                      [4.0, 40.0]])
    """
    # flatten the list of global indices
    all_idx = torch.cat(idx_out, dim=0)

    x_list = []
    for g, g_idx in enumerate(idx_out):
        x_list.append(x_graph[g, : g_idx.size(0), :])
    x_cat = torch.cat(x_list, dim=0)  # [N_total, D]

    return x_cat[torch.argsort(all_idx)]


@torch.jit.ignore
def find_connected_components_undirected(
    num_nodes: int, edge_index: torch.LongTensor
) -> List[torch.Tensor]:
    """
    Find all connected components in an undirected graph using DFS.

    Parameters
    ----------
    num_nodes: int
        number of possible nodes (labels range 0..num_nodes-1)
    edge_index: torch.LongTensor
        shape [2, E], edges assumed undirected

    Returns
    -------
    components: List[torch.Tensor]
        each is a list of node indices belonging to a connected component
        ONLY nodes that appear in edge_index are included.
    """
    if edge_index.numel() == 0:
        return []  # no edges → no components per your earlier rule

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    E = edge_index.size(1)

    # Build adjacency list (undirected)
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        adj[u].append(v)
        adj[v].append(u)  # <-- critical: add reverse edge

    visited = [False] * num_nodes
    components = []

    # Nodes that are actually present in the graph
    graph_nodes = set(src) | set(dst)

    # Standard DFS
    def dfs(start):
        stack = [start]
        comp = []
        visited[start] = True

        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        return comp

    # Launch DFS from nodes that actually exist in edge_index
    for u in sorted(graph_nodes):
        if not visited[u]:
            comp = dfs(u)
            components.append(torch.tensor(comp, dtype=torch.long))

    return components


def find_local_edge_index(
    edge_index: torch.LongTensor, batch: torch.LongTensor, b: int
):
    """
    Find edge_index belonging to graph b in the batch. Local node indices are returned.

    Parameters
    ----------
    edge_index: torch.LongTensor
        edge index, shape [2, E], where the first row indicates source global nodes indices, and the second row indicates destination global node indices
    batch: torch.LongTensor
        batch vector, shape [N], batch[i] = graph index of node i
    b: int
        desired graph index in the batch

    Returns
    -------
    edge_index_b: torch.LongTensor
        edge index of graph b, shape [2, E_b]
    """
    node_mask = batch == b
    if not node_mask.any():
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    num_nodes_b = node_mask.sum().item()
    node_map = torch.full(
        (batch.size(0),), -1, dtype=torch.long, device=edge_index.device
    )
    node_map[node_mask] = torch.arange(num_nodes_b, device=edge_index.device)

    src = edge_index[0]
    dst = edge_index[1]
    mask = node_mask[src] & node_mask[dst]
    edge_index_b = edge_index[:, mask]

    # map global → local
    return node_map[edge_index_b]


def create_unique_object_ids(
    object_ids: torch.Tensor,
    batch: torch.Tensor,
    noise_idx: int = -1,
) -> torch.Tensor:
    """
    Create unique object IDs across batches, preserving noise labels.

    Parameters
    ----------
    object_ids: torch.Tensor
        shape [N], object ID for each node, where N is total number of nodes across all graphs. Noise nodes have IDs in noise_idx.
    batch: torch.Tensor
        shape [N], batch vector, batch[i] = graph index of node i
    noise_idx: int, optional
        noise node ID, default -1

    Returns
    -------
    new_object_ids: torch.Tensor
        shape [N], new object IDs for each node, where IDs are unique across batches for signal nodes, and noise nodes retain their original IDs.

    Examples
    --------
    >>> batch = torch.tensor([0, 0, 0, 1, 1, 2])
    >>> object_ids = torch.tensor([0, 1, -1, 0, 1, -1])
    >>> noise_idx = -1
    >>> new_object_ids = create_unique_object_ids(object_ids, batch, noise_idx)
    >>> print(new_object_ids)
    >>> torch.tensor([0, 1, -1, 2, 3, -1])
    """
    is_signal = object_ids != noise_idx

    if is_signal.sum() == 0:
        return object_ids.clone()

    result = object_ids.clone()

    # Get unique (batch, object_id) pairs for signal nodes
    signal_mask = is_signal
    signal_batch = batch[signal_mask]
    signal_oid = object_ids[signal_mask]

    # Create compound keys and get unique pairs
    max_oid = signal_oid.max().item() + 1
    compound_key = signal_batch * max_oid + signal_oid
    _, inverse = torch.unique(compound_key, sorted=True, return_inverse=True)

    # Assign new sequential IDs (starting after max noise index)
    new_ids = inverse + noise_idx + 1
    result[signal_mask] = new_ids

    return result
