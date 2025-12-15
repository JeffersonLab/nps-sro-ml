import torch
from typing import Optional

try:
    from torch_scatter import scatter_max, scatter_add, scatter_mean
except ImportError:
    raise ImportError(
        "torch_scatter is required for oc_loss.py. Please install `torch-scatter`, see `https://github.com/rusty1s/pytorch_scatter` for instructions."
    )


def oc_loss_per_batch(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    q_min: float = 0.1,
    noise_idx: int | list[int] = 0,
    margin: float = 1.0,
    batch: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute all four Object Condensation losses per batch in a fully-vectorized manner. This function assumes that object IDs, except background, are unique across different graphs in the batch. User should modify the object IDs accordingly before passing to this function.

    Parameters
    ----------
    x : torch.Tensor
        Latent space positions of shape [num_nodes, pos_dim].
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes]. Object IDs from different graphs should be unique, except for background.
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.
    noise_idx : int | list[int], optional
        Index or list of indices representing noise/background points in object_id, by default 0.
    margin : float, optional
        Margin distance for repulsive potential, by default 1.0. Only points within this distance contribute to the loss.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Attractive potential loss, repulsive potential loss, cowardice penalty loss, noise penalty loss.

    """
    # TODO : construct unique object IDs across batch, except for background

    attr_loss = oc_attr_loss_per_batch(x, beta, object_id, q_min, noise_idx, batch)
    repul_loss = oc_repul_loss_per_batch(
        x, beta, object_id, q_min, noise_idx, margin, batch
    )
    coward_loss = oc_coward_loss_per_batch(beta, object_id, noise_idx, batch)
    noise_loss = oc_noise_loss_per_batch(beta, object_id, noise_idx, batch)
    return attr_loss, repul_loss, coward_loss, noise_loss


def oc_attr_loss_per_batch(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    q_min: float = 0.1,
    noise_idx: int | list[int] = 0,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the attractive loss for Object Condensation per batch in a fully-vectorized manner. Definition of the loss can be found in the docstring of `oc_attr_loss_per_graph`. The losses from each graph in the batch are summed at the end. Currently, this function assumes that object IDs, except background, are unique across different graphs in the batch. User should modify the object IDs accordingly before passing to this function.

    Parameters
    ----------
    x : torch.Tensor
        Latent space positions of shape [num_nodes, pos_dim].
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes]. Object IDs from different graphs should be unique, except for background.
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.
    noise_idx : int | list[int], optional
        Index or list of indices representing noise/background points in object_id, by default 0.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.

    Returns
    -------
    torch.Tensor
        Attractive potential loss.
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    beta = torch.clamp(beta, max=0.9999)
    q = torch.arctanh(beta) ** 2 + q_min
    is_sig = ~torch.isin(object_id, torch.tensor(noise_idx, device=object_id.device))

    if is_sig.sum() == 0:
        return x.sum() * 0.0

    # find representative q for each object, excluding background
    sig_indices = torch.where(is_sig)[0]  # global indices of signals
    # get a array of group ids from 0 to n_objs-1 for each signal point
    unique_oid, obj_gp_id = torch.unique(object_id[is_sig], return_inverse=True)
    # for each object, get the max q and argmax within the signal points
    q_repr, obj_id_repr = scatter_max(q[is_sig], obj_gp_id, dim_size=unique_oid.size(0))
    # this id is relative to signal points, so we need to map it back to global indices
    id_repr = sig_indices[obj_id_repr]  # global indices of representatives

    # calculate distances between all points and object representatives
    dist_jk = torch.cdist(x, x[id_repr])
    q_jk = q.view(-1, 1) * q_repr.view(1, -1)
    batch_idx = batch.view(-1, 1).expand(-1, id_repr.size(0))

    # attractive potential
    attr_mask = object_id.view(-1, 1) == object_id[id_repr].view(1, -1)
    obj_sizes = attr_mask.sum(dim=0).clamp(min=1).view(1, -1)
    v_attr = torch.square(dist_jk) * q_jk / obj_sizes
    l_attr = scatter_add(
        v_attr[attr_mask], batch_idx[attr_mask], dim_size=batch.unique().size(0)
    )

    # number of objects (excluding singletons) in each graph
    # the order is automatically aligned with l_attr since the same `batch` is used
    large_obj = (obj_sizes > 1).float()
    n_obj_per_graph = scatter_add(
        large_obj, batch[id_repr], dim_size=batch.unique().size(0)
    )
    return (l_attr / n_obj_per_graph.clamp(min=1)).sum()


def oc_repul_loss_per_batch(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    q_min: float = 0.1,
    noise_idx: int | list[int] = 0,
    margin: float = 1.0,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the repulsive loss for Object Condensation per batch in a fully-vectorized manner. Definition of the loss can be found in the docstring of `oc_repul_loss_per_graph`. The losses from each graph in the batch are summed at the end. Currently, this function assumes that object IDs, except background, are unique across different graphs in the batch. User should modify the object IDs accordingly before passing to this function.

    Parameters
    ----------
    x : torch.Tensor
        Latent space positions of shape [num_nodes, pos_dim].
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes]. Object IDs from different graphs should be unique, except for background.
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.
    noise_idx : int | list[int], optional
        Index or list of indices representing noise/background points in object_id, by default 0.
    margin : float, optional
        Margin distance for repulsive potential, by default 1.0. Only points within this distance contribute to the loss.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.

    Returns
    -------
    torch.Tensor
        Repulsive potential loss.
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    beta = torch.clamp(beta, max=0.9999)
    q = torch.arctanh(beta) ** 2 + q_min
    is_sig = ~torch.isin(object_id, torch.tensor(noise_idx, device=object_id.device))

    if is_sig.sum() == 0:
        return x.sum() * 0.0

    # find representative q for each object, excluding background
    sig_indices = torch.where(is_sig)[0]  # global indices of signals
    # get a array of group ids from 0 to n_objs-1 for each signal point
    unique_oid, obj_gp_id = torch.unique(object_id[is_sig], return_inverse=True)
    # for each object, get the max q and argmax within the signal points
    q_repr, obj_id_repr = scatter_max(q[is_sig], obj_gp_id, dim_size=unique_oid.size(0))
    # this id is relative to signal points, so we need to map it back to global indices
    id_repr = sig_indices[obj_id_repr]  # global indices of representatives

    # calculate distances between all points and object representatives
    dist_jk = torch.cdist(x, x[id_repr])
    q_jk = q.view(-1, 1) * q_repr.view(1, -1)
    batch_idx = batch.view(-1, 1).expand(-1, id_repr.size(0))

    # repulsive potential (between obj and all points not in obj, i.e. other objs + background)
    same_batch = batch.view(-1, 1) == batch[id_repr].view(1, -1)
    attr_mask = object_id.view(-1, 1) == object_id[id_repr].view(1, -1)
    repul_mask = (~attr_mask) & (dist_jk < margin) & same_batch

    # calculate the repulsive norm per batch = number of repulsive candidates
    repul_norm = (~attr_mask & same_batch).sum(dim=0).clamp(min=1).view(1, -1)

    v_repul = (margin - dist_jk) * q_jk / repul_norm
    l_repul = scatter_add(
        v_repul[repul_mask], batch_idx[repul_mask], dim_size=batch.unique().size(0)
    )

    # number of objects in each graph
    # the order is automatically aligned with l_repul since the same `batch` is used
    n_obj_per_graph = scatter_add(
        torch.ones_like(q_repr), batch[id_repr], dim_size=batch.unique().size(0)
    )

    return (l_repul / n_obj_per_graph.clamp(min=1)).sum()


def oc_coward_loss_per_batch(
    beta: torch.Tensor,
    object_id: torch.Tensor,
    noise_idx: int | list[int] = 0,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the cowardice penalty loss for Object Condensation per batch in a fully-vectorized manner. Mathematically, this is the mean (1 - beta) value of all object representatives per graph, then summed over all graphs.

    Parameters
    ----------
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes].
    noise_idx : int | list[int], optional
        Index or list of indices representing noise/background points in object_id, by default 0.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.

    Returns
    -------
    torch.Tensor
        Cowardice penalty loss.
    """
    if batch is None:
        batch = torch.zeros(beta.size(0), dtype=torch.long, device=beta.device)

    if isinstance(noise_idx, int):
        noise_idx = [noise_idx]

    is_sig = ~torch.isin(object_id, torch.tensor(noise_idx, device=object_id.device))

    if is_sig.sum() == 0:
        return beta.sum() * 0.0

    # find representative q for each object, excluding background
    sig_indices = torch.where(is_sig)[0]  # global indices of signals
    # get a array of group ids from 0 to n_objs-1 for each signal point
    unique_oid, obj_gp_id = torch.unique(object_id[is_sig], return_inverse=True)
    # for each object, get the max beta and argmax within the signal points
    beta_repr, obj_id_repr = scatter_max(
        beta[is_sig], obj_gp_id, dim_size=unique_oid.size(0)
    )
    # this id is relative to signal points, so we need to map it back to global indices
    id_repr = sig_indices[obj_id_repr]  # global indices of representatives

    l_coward = scatter_mean(
        1 - beta[id_repr], batch[id_repr], dim_size=batch.unique().size(0)
    )
    return l_coward.sum()


def oc_noise_loss_per_batch(
    beta: torch.Tensor,
    object_id: torch.Tensor,
    noise_idx: int | list[int] = 0,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the noise penalty loss for Object Condensation per batch in a fully-vectorized manner. Mathematically, this is the mean beta value of all noise/background points per graph, then summed over all graphs.

    Parameters
    ----------
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes].
    noise_idx : int | list[int], optional
        Index or list of indices representing noise/background points in object_id, by default 0.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.

    Returns
    -------
    torch.Tensor
        Noise penalty loss.
    """
    if batch is None:
        batch = torch.zeros(beta.size(0), dtype=torch.long, device=beta.device)

    if isinstance(noise_idx, int):
        noise_idx = [noise_idx]

    is_noise = torch.isin(object_id, torch.tensor(noise_idx, device=object_id.device))
    if is_noise.sum() == 0:
        return beta.sum() * 0.0

    l_noise = scatter_mean(
        beta[is_noise], batch[is_noise], dim_size=batch.unique().size(0)
    )
    return l_noise.sum()


def oc_attr_loss_per_graph(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    q_min: float = 0.1,
    noise_idx: int | list[int] = 0,
):
    """
    Compute the attractive loss for Object Condensation per graph. For each object, the loss is computed between its representative point and all other points belonging to the same object. Each pair of points is defined as L2 distance squared weighted by their charges, normalized by the number of points in the object. The final loss is averaged over all objects in the graph.

    Parameters
    ----------
    x : torch.Tensor
        Latent space positions of shape [num_nodes, pos_dim].
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes].
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.
    noise_idx : int | list[int], optional
        Index or list of indices representing noise/background points in object_id, by default 0.

    Returns
    -------
    torch.Tensor
        Attractive potential loss.

    """
    if isinstance(noise_idx, int):
        noise_idx = [noise_idx]

    q = torch.arctanh(beta) ** 2 + q_min  # shape [num_nodes]
    is_sig = ~torch.isin(object_id, torch.tensor(noise_idx, device=object_id.device))

    if is_sig.sum() == 0:
        return x.sum() * 0.0

    unique_obj_ids = torch.unique(object_id[is_sig])
    attr_mask = object_id.view(-1, 1) == unique_obj_ids.view(1, -1)
    obj_sizes = attr_mask.sum(dim=0).clamp(min=1)  # avoid division by zero
    alphas = torch.argmax(q.view(-1, 1) * attr_mask, dim=0)  # shape [num_objects]

    dist_jk = torch.cdist(x, x[alphas])  # shape [num_nodes, num_objects]
    q_jk = q.view(-1, 1) * q[alphas].view(1, -1)  # shape [num_nodes, num_objects]
    q_jk = q_jk / obj_sizes  # normalize by object size

    loss = (q_jk * torch.square(dist_jk))[attr_mask].sum()

    large_obj_mask = obj_sizes > 1
    loss = loss / large_obj_mask.sum().clamp(
        min=1
    )  # average over number of large objects
    return loss
    # return loss / (len(unique_obj_ids) if len(unique_obj_ids) > 0 else 1)


def oc_repul_loss_per_graph(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    q_min: float = 0.1,
    noise_idx: int | list[int] = 0,
    margin: float = 1.0,
):
    """
    Compute the repulsive loss for Object Condensation per graph. For each object, the loss is computed between its representative point and all other points not belonging to the same object, within a specified margin. Each pair of points is defined as L2 distance weighted by their charges, normalized by the number of all possible repulsive points regardless of margin. The final loss is averaged over all objects in the graph.

    Parameters
    ----------
    x : torch.Tensor
        Latent space positions of shape [num_nodes, pos_dim].
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes].
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.
    noise_idx : int | list[int], optional
        Index or list of indices representing noise/background points in object_id, by default 0.
    margin : float, optional
        Margin distance for repulsive potential, by default 1.0. Only points within this distance contribute to the loss.

    Returns
    -------
    torch.Tensor
        Repulsive potential loss.
    """
    if isinstance(noise_idx, int):
        noise_idx = [noise_idx]

    q = torch.arctanh(beta) ** 2 + q_min  # shape [num_nodes]
    is_sig = ~torch.isin(object_id, torch.tensor(noise_idx, device=object_id.device))

    if is_sig.sum() == 0:
        return x.sum() * 0.0

    unique_obj_ids = torch.unique(object_id[is_sig])
    attr_mask = object_id.view(-1, 1) == unique_obj_ids.view(
        1, -1
    )  # shape [num_nodes, num_objects]
    alphas = torch.argmax(q.view(-1, 1) * attr_mask, dim=0)  # shape [num_objects]

    dist_jk = torch.cdist(x, x[alphas])  # shape [num_nodes, num_objects]
    q_jk = q.view(-1, 1) * q[alphas].view(1, -1)  # shape [num_nodes, num_objects]

    repul_mask = (~attr_mask) & (dist_jk < margin)
    repul_norm = (~attr_mask).sum(dim=0).clamp(min=1)  # avoid no nearby points
    q_jk = q_jk / repul_norm  # normalize by number of repulsive points

    loss = ((margin - dist_jk)[repul_mask] * q_jk[repul_mask]).sum()
    return loss / len(unique_obj_ids)


def oc_coward_loss_per_graph(
    beta: torch.Tensor,
    object_id: torch.Tensor,
    noise_idx: int | list[int] = 0,
):
    """
    Compute the cowardice penalty loss for Object Condensation per graph. Mathematically, this is the mean (1 - beta) value of all object representatives.

    Parameters
    ----------
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes].
    noise_idx : int | list[int], optional
        Index or list of indices representing noise/background points in object_id, by default 0.

    Returns
    -------
    torch.Tensor
        Cowardice penalty loss.
    """
    if isinstance(noise_idx, int):
        noise_idx = [noise_idx]

    is_sig = ~torch.isin(object_id, torch.tensor(noise_idx, device=object_id.device))

    if is_sig.sum() == 0:
        return beta.sum() * 0.0

    unique_obj_ids = torch.unique(object_id[is_sig])
    attr_mask = object_id.view(-1, 1) == unique_obj_ids.view(1, -1)
    alphas = torch.argmax(beta.view(-1, 1) * attr_mask, dim=0)  # shape [num_objects]

    loss = torch.mean(1 - beta[alphas])
    return loss


def oc_noise_loss_per_graph(
    beta: torch.Tensor,
    object_id: torch.Tensor,
    noise_idx: int | list[int] = 0,
):
    """
    Compute the noise penalty loss for Object Condensation per graph. Mathematically, this is the mean beta value of all noise/background points.

    Parameters
    ----------
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes].
    noise_idx : int | list[int], optional
        Index or list of indices representing noise/background points in object_id, by default 0.

    Returns
    -------
    torch.Tensor
        Noise penalty loss.
    """
    if isinstance(noise_idx, int):
        noise_idx = [noise_idx]

    is_noise = torch.isin(object_id, torch.tensor(noise_idx, device=object_id.device))
    if is_noise.sum() == 0:
        return beta.sum() * 0.0

    return torch.mean(beta[is_noise])


def oc_attr_loss_per_graph_naive(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    q_min: float = 0.1,
    noise_idx: int | list[int] = 0,
):
    """
    Naive implementation of attractive loss per graph for Object Condensation. Loss is accumulated by iterating over each unique object ID. The purpose of the function is for testing and validation of the vectorized version only. Do not use in backpropagation of gradients.

    Parameters
    ----------
    x : torch.Tensor
        Latent space positions of shape [num_nodes, pos_dim].
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes].
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.
    noise_idx : int | list[int], optional
        Index or list of indices representing noise/background points in object_id, by default 0.

    Returns
    -------
    torch.Tensor
        Attractive potential loss.
    """
    if x.requires_grad:
        raise RuntimeError(
            "This is a naive implementation for testing only. Do not use in backpropagation of gradients."
        )

    if isinstance(noise_idx, int):
        noise_idx = [noise_idx]

    q = torch.arctanh(beta) ** 2 + q_min  # shape [num_nodes]
    loss = x.sum() * 0.0

    is_sig = ~torch.isin(object_id, torch.tensor(noise_idx, device=object_id.device))
    unique_obj_ids = torch.unique(object_id[is_sig])

    if len(unique_obj_ids) == 0:
        return loss

    large_obj_count = 0
    for obj in unique_obj_ids:
        mask = object_id == obj
        x_obj, q_obj = x[mask], q[mask]
        cid = q_obj.argmax()
        x_repr, q_repr = x_obj[cid], q_obj[cid]
        obj_size = torch.sum(mask).clamp(min=1)
        if obj_size > 1:
            large_obj_count += 1
        dist = torch.cdist(x_obj, x_repr.unsqueeze(0), p=2).squeeze()
        loss += torch.sum(q_obj * q_repr * torch.square(dist)) / obj_size

    return loss / large_obj_count if large_obj_count > 0 else loss


def oc_repul_loss_per_graph_naive(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    q_min: float = 0.1,
    noise_idx: int | list[int] = 0,
    margin: float = 1.0,
):
    """
    Naive implementation of repulsive loss per graph for Object Condensation. Loss is accumulated by iterating over each unique object ID. The purpose of the function is for testing and validation of the vectorized version only. Do not use in backpropagation of gradients.

    Parameters
    ----------
    x : torch.Tensor
        Latent space positions of shape [num_nodes, pos_dim].
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes].
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.
    noise_idx : int | list[int], optional
        Index or list of indices representing noise/background points in object_id, by default 0.
    margin : float, optional
        Margin distance for repulsive potential, by default 1.0.

    Returns
    -------
    torch.Tensor
        Repulsive potential loss.
    """
    if x.requires_grad:
        raise RuntimeError(
            "This is a naive implementation for testing only. Do not use in backpropagation of gradients."
        )

    if isinstance(noise_idx, int):
        noise_idx = [noise_idx]

    beta = torch.clamp(beta, max=0.9999)
    q = torch.arctanh(beta) ** 2 + q_min  # shape [num_nodes]
    loss = x.sum() * 0.0

    is_sig = ~torch.isin(object_id, torch.tensor(noise_idx, device=object_id.device))
    unique_obj_ids = torch.unique(object_id[is_sig])

    if len(unique_obj_ids) == 0:
        return loss

    for obj in unique_obj_ids:
        mask = object_id == obj
        x_obj, q_obj = x[mask], q[mask]
        cid = q_obj.argmax()
        x_repr, q_repr = x_obj[cid], q_obj[cid]

        dist = torch.cdist(x, x_repr.unsqueeze(0), p=2).squeeze()
        repul_mask = (object_id != obj) & (dist < margin)
        n_repul = (object_id != obj).sum().clamp(min=1)
        loss += (
            torch.sum(q[repul_mask] * q_repr * (margin - dist[repul_mask])) / n_repul
        )
    return loss / len(unique_obj_ids)
