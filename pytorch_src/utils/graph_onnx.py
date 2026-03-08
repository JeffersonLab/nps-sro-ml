import torch
from typing import List, Tuple


def pack_to_graph_batches_onnx(
    x: torch.Tensor, t: List[torch.Tensor], batch: torch.LongTensor, B: int, L_max: int
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.BoolTensor]:
    """
    Pack node features and edge features into graph batches. For Onnx compatibility, the max number of graphs (B) and max number of nodes (L_max) must be specified, and all graphs will be padded to these dimensions.

    Parameters
    ----------
    x: torch.Tensor
        Node features, shape [N_total, D], where N_total is total number of nodes across all graphs, D is feature dimension.
    t: List[torch.Tensor]
        List of additional node-level tensors to pack, each of shape [N_total, D_t], where D_t is the feature dimension of that tensor.
    batch: torch.LongTensor
        Batch vector, shape [N_total], batch[i] = graph index of node i.
    B : int
        Maximum number of graphs per batch.
    L_max : int
        Maximum number of nodes in any graph in the batch. All graphs will be padded to this length.

    Returns
    -------
    outs: List[torch.Tensor]
        List of node feature tensors in graph-batched format. The first element is the packed x tensor of shape [B, L_max, D].
    idx_out: torch.Tensor
        global index tensors for each graph in the batch.
    mask_out: torch.BoolTensor
        Mask tensor indicating valid nodes in the graph-batched format, shape [B, L_max].

    """

    device = x.device
    N, D = x.size()

    arange_N = torch.arange(N, device=device)
    arange_B = torch.arange(B, device=device)

    # [N, B] one-hot, used for both counts and position assignment
    one_hot = (batch.unsqueeze(1) == arange_B.unsqueeze(0)).long()  # [N, B]
    counts = one_hot.sum(dim=0)  # number of nodes per graph, [B]

    # Within-graph position for each node
    pos_in_graph = (one_hot.cumsum(dim=0) - 1)[arange_N, batch]  # [N]

    # Packed features
    outs: List[torch.Tensor] = []

    out_x = torch.zeros((B, L_max, D), device=device, dtype=x.dtype)
    out_x[batch, pos_in_graph] = x
    outs.append(out_x)

    for t_ in t:
        _, D_t = t_.size()
        out_t = torch.zeros((B, L_max, D_t), device=device, dtype=t_.dtype)
        out_t[batch, pos_in_graph] = t_
        outs.append(out_t)

    # Mask
    pos_range = torch.arange(L_max, device=device).unsqueeze(0)  # [1, L_max]
    mask_out = pos_range < counts.unsqueeze(1)  # [B, L_max]

    # idx_out as padded [B, L_max] tensor
    idx_out = torch.full((B, L_max), fill_value=-1, dtype=torch.long, device=device)
    idx_out[batch, pos_in_graph] = arange_N

    return outs, idx_out, mask_out


def reorder_from_graph_batches_onnx(
    x_graph: torch.Tensor,
    idx_out: torch.Tensor,
    N_total: int,
) -> torch.Tensor:
    """
    Reorder node features from graph-batched format back to original node order. For compatibility with Onnx, total number of valid nodes (N_total) must be specified.

    Parameters
    ----------
    x_graph: torch.Tensor
        Node features in graph-batched format, shape [B, L_max, D], where B is batch size, L_max is max number of nodes per graph, D is feature dimension.
    idx_out: torch.Tensor
        Padded index tensor for each graph in the batch, shape [B, L_max].
    N_total: int
        Total number of valid nodes across all graphs in the batch.

    Returns
    -------
    x: torch.Tensor
        Node features reordered to original node order, shape [N_total, D], where N_total is total number of nodes across all graphs. Note that graphs with fewer than L_max nodes are truncated to their actual lengths.
    """

    B, L_max, D = x_graph.shape

    x_flat = x_graph.reshape(B * L_max, D)
    idx_flat = idx_out.reshape(B * L_max)

    valid_positions = (idx_flat >= 0).nonzero(as_tuple=False).squeeze(1)
    x_valid = x_flat[valid_positions]
    idx_valid = idx_flat[valid_positions]

    x_out = torch.zeros((N_total, D), device=x_graph.device, dtype=x_graph.dtype)
    x_out[idx_valid] = x_valid

    return x_out
