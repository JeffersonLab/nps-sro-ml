import torch


def structural_causal_mask(B: int, L: int, device: torch.device) -> torch.Tensor:
    mask_shape = [B, 1, L, L]
    mask = torch.triu(
        torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1
    )
    return mask


def cross_graph_mask(batch: torch.Tensor) -> torch.BoolTensor:
    """
    Create a block diagonal mask for a batch of graphs, where each graph is represented by its nodes. The mask will have True values for pairs of nodes that belong to different graphs and False values for pairs of nodes that belong to the same graph.

    Parameters
    ----------
    batch: torch.Tensor
        Batch vector, shape [N], batch[i] = graph index of node i

    Returns
    -------
    mask: torch.BoolTensor
        Mask tensor, shape [N, N], where mask[i, j] = True if nodes i and j belong to different graphs, and False otherwise.
    """
    graph_ids_q = batch.reshape(1, 1, -1, 1)
    graph_ids_k = batch.reshape(1, 1, 1, -1)
    mask = graph_ids_q != graph_ids_k
    return mask
