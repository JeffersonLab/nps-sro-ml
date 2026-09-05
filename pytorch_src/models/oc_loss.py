import torch

import warnings
from typing import Optional

try:
    from torch_scatter import scatter_max, scatter_add, scatter_mean

    _HAS_TORCH_SCATTER = True
except ImportError:
    _HAS_TORCH_SCATTER = False


def _sum_loss_per_graph(
    loss_fn: callable,
    batch: Optional[torch.Tensor],
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    Sum the loss from each graph in the batch. This function assumes that the loss function `loss_fn` takes the same arguments as the batch-level loss functions defined in this module, except for the `batch` argument.

    Parameters
    ----------
    loss_fn : callable
        Loss function to compute the loss for each graph.
    batch : Optional[torch.Tensor]
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.
    *args : tuple
        Positional arguments to be passed to the loss function.
    **kwargs : dict
        Keyword arguments to be passed to the loss function.

    Returns
    -------
    torch.Tensor
        Total loss summed over all graphs in the batch.
    """
    if batch is None:
        return loss_fn(*args, **kwargs)

    loss = None
    for graph_id in torch.unique(batch, sorted=True):
        mask = batch == graph_id
        graph_args = [
            arg[mask] if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        kwargs_ = {
            k: v[mask] if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }
        graph_loss = loss_fn(*graph_args, **kwargs_)
        loss = graph_loss if loss is None else loss + graph_loss
    return loss


def oc_loss_per_batch(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    feat_loss: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
    margin: float = 1.0,
    use_scatter: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
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
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects, by default None.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.
    feat_loss : Optional[torch.Tensor], optional
        Per-node feature loss of shape [num_nodes]. If None, the feature loss is not computed, by default None.
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.
    margin : float, optional
        Margin distance for repulsive potential, by default 1.0. Only points within this distance contribute to the loss.
    use_scatter : bool, optional
        Whether to use torch_scatter for computation. If False, a fallback implementation is used, by default False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        Attractive potential loss, repulsive potential loss, cowardice penalty loss, noise penalty loss, feature loss.

    """
    if use_scatter and _HAS_TORCH_SCATTER is False:
        warnings.warn(
            "torch_scatter is not installed. Falling back to a per graph loop implementation. This may affect performance."
        )
        use_scatter = False

    attr_loss = oc_attr_loss_per_batch(
        x, beta, object_id, is_sig, batch, q_min=q_min, use_scatter=use_scatter
    )
    repul_loss = oc_repul_loss_per_batch(
        x,
        beta,
        object_id,
        is_sig,
        q_min=q_min,
        margin=margin,
        batch=batch,
        use_scatter=use_scatter,
    )
    coward_loss = oc_coward_loss_per_batch(
        beta, object_id, is_sig=is_sig, batch=batch, use_scatter=use_scatter
    )
    noise_loss = oc_noise_loss_per_batch(
        beta, is_sig=is_sig, batch=batch, use_scatter=use_scatter
    )

    if feat_loss is not None:
        feat_loss = oc_feat_loss_per_batch(
            feat_loss,
            beta,
            object_id,
            is_sig,
            batch,
            q_min=q_min,
            use_scatter=use_scatter,
        )
    else:
        feat_loss = None

    return attr_loss, repul_loss, coward_loss, noise_loss, feat_loss


def oc_attr_loss_per_batch(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
    use_scatter: bool = False,
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
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects, by default None.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.
    use_scatter : bool, optional
        Whether to use torch_scatter for computation. If False, a fallback implementation is used, by default False.

    Returns
    -------
    torch.Tensor
        Attractive potential loss.
    """
    if use_scatter and _HAS_TORCH_SCATTER is False:
        warnings.warn(
            "torch_scatter is not installed. Falling back to a per graph loop implementation. This may affect performance."
        )
        use_scatter = False

    if use_scatter:
        return _oc_attr_loss_per_batch_scatter(x, beta, object_id, is_sig, batch, q_min)
    else:
        return _sum_loss_per_graph(
            oc_attr_loss_per_graph,
            batch,
            x,
            beta,
            object_id,
            is_sig,
            q_min=q_min,
        )


def oc_repul_loss_per_batch(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
    margin: float = 1.0,
    use_scatter: bool = False,
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
    is_sig : torch.Tensor, optional
        Boolean mask indicating significant points (True) and noise/background points (False), by default None.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.
    margin : float, optional
        Margin distance for repulsive potential, by default 1.0. Only points within this distance contribute to the loss.
    use_scatter : bool, optional
        Whether to use torch_scatter for computation. If False, a fallback implementation is used, by default False.

    Returns
    -------
    torch.Tensor
        Repulsive potential loss.
    """
    if use_scatter and _HAS_TORCH_SCATTER is False:
        warnings.warn(
            "torch_scatter is not installed. Falling back to a per graph loop implementation. This may affect performance."
        )
        use_scatter = False

    if use_scatter:
        return _oc_repul_loss_per_batch_scatter(
            x, beta, object_id, is_sig, q_min, margin, batch
        )
    else:
        return _sum_loss_per_graph(
            oc_repul_loss_per_graph,
            batch,
            x,
            beta,
            object_id,
            is_sig=is_sig,
            q_min=q_min,
            margin=margin,
        )


def oc_coward_loss_per_batch(
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    use_scatter: bool = False,
) -> torch.Tensor:
    """
    Compute the cowardice penalty loss for Object Condensation per batch in a fully-vectorized manner. Mathematically, this is the mean (1 - beta) value of all object representatives per graph, then summed over all graphs.

    Parameters
    ----------
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes].
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects, by default None.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.
    use_scatter : bool, optional
        Whether to use torch_scatter for computation. If False, a fallback implementation is used, by default False.

    Returns
    -------
    torch.Tensor
        Cowardice penalty loss.
    """
    if use_scatter and _HAS_TORCH_SCATTER is False:
        warnings.warn(
            "torch_scatter is not installed. Falling back to a per graph loop implementation. This may affect performance."
        )
        use_scatter = False

    if use_scatter:
        return _oc_coward_loss_per_batch_scatter(beta, object_id, is_sig, batch)
    else:
        return _sum_loss_per_graph(
            oc_coward_loss_per_graph,
            batch,
            beta,
            object_id,
            is_sig,
        )


def oc_noise_loss_per_batch(
    beta: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    use_scatter: bool = False,
) -> torch.Tensor:
    """
    Compute the noise penalty loss for Object Condensation per batch in a fully-vectorized manner. Mathematically, this is the mean beta value of all noise/background points per graph, then summed over all graphs.

    Parameters
    ----------
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects, by default None.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.

    Returns
    -------
    torch.Tensor
        Noise penalty loss.
    """

    if use_scatter and _HAS_TORCH_SCATTER is False:
        warnings.warn(
            "torch_scatter is not installed. Falling back to a per graph loop implementation. This may affect performance."
        )
        use_scatter = False

    if use_scatter:
        return _oc_noise_loss_per_batch_scatter(beta, is_sig, batch)
    else:
        return _sum_loss_per_graph(
            oc_noise_loss_per_graph,
            batch,
            beta,
            is_sig,
        )


def oc_feat_loss_per_batch(
    feat_loss: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
    use_scatter: bool = False,
) -> torch.Tensor:

    if use_scatter and _HAS_TORCH_SCATTER is False:
        warnings.warn(
            "torch_scatter is not installed. Falling back to a per graph loop implementation. This may affect performance."
        )
        use_scatter = False

    if use_scatter:
        _oc_feat_loss_per_batch_scatter(
            feat_loss, beta, object_id, is_sig, batch, q_min
        )
    else:
        return _sum_loss_per_graph(
            oc_feat_loss_per_graph,
            batch,
            feat_loss,
            beta,
            is_sig,
            q_min,
        )


def _oc_attr_loss_per_batch_scatter(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
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
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects, by default None.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.

    Returns
    -------
    torch.Tensor
        Attractive potential loss.
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    beta = torch.clamp(beta, max=0.9999)
    q = torch.arctanh(beta) ** 2 + q_min
    if is_sig is None:
        is_sig = torch.ones_like(beta, dtype=torch.bool)

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


def _oc_repul_loss_per_batch_scatter(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
    margin: float = 1.0,
    batch: Optional[torch.Tensor] = None,
):
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    beta = torch.clamp(beta, max=0.9999)
    q = torch.arctanh(beta) ** 2 + q_min

    if is_sig is None:
        is_sig = torch.ones_like(beta, dtype=torch.bool)
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


def _oc_coward_loss_per_batch_scatter(
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
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
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects.
    batch : Optional[torch.Tensor], optional
        Graph indices for each node, by default None. If None, all nodes are considered to belong to a single graph.

    Returns
    -------
    torch.Tensor
        Cowardice penalty loss.
    """
    if batch is None:
        batch = torch.zeros(beta.size(0), dtype=torch.long, device=beta.device)

    if is_sig is None:
        is_sig = torch.ones_like(beta, dtype=torch.bool)

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


def _oc_noise_loss_per_batch_scatter(
    beta: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    if batch is None:
        batch = torch.zeros(beta.size(0), dtype=torch.long, device=beta.device)

    if is_sig is None:
        is_sig = torch.ones_like(beta, dtype=torch.bool)

    is_noise = ~is_sig
    if is_noise.sum() == 0:
        return beta.sum() * 0.0

    l_noise = scatter_mean(
        beta[is_noise], batch[is_noise], dim_size=batch.unique().size(0)
    )
    return l_noise.sum()


def _oc_feat_loss_per_batch_scatter(
    feat_loss: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
):
    raise NotImplementedError(
        "oc_feat_loss_per_batch with use_scatter=True is not implemented yet."
    )


def oc_attr_loss_per_graph(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
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
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects, by default None.
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.

    Returns
    -------
    torch.Tensor
        Attractive potential loss.

    """
    q = torch.arctanh(beta) ** 2 + q_min  # shape [num_nodes]

    if is_sig is None:
        is_sig = torch.ones_like(beta, dtype=torch.bool)
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


def oc_repul_loss_per_graph(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
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
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects, by default None.
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.
    margin : float, optional
        Margin distance for repulsive potential, by default 1.0. Only points within this distance contribute to the loss.

    Returns
    -------
    torch.Tensor
        Repulsive potential loss.
    """
    q = torch.arctanh(beta) ** 2 + q_min  # shape [num_nodes]

    if is_sig is None:
        is_sig = torch.ones_like(beta, dtype=torch.bool)

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
    is_sig: Optional[torch.Tensor] = None,
):
    """
    Compute the cowardice penalty loss for Object Condensation per graph. Mathematically, this is the mean (1 - beta) value of all object representatives.

    Parameters
    ----------
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    object_id : torch.Tensor
        Ground truth object IDs of shape [num_nodes].
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects, by default None.

    Returns
    -------
    torch.Tensor
        Cowardice penalty loss.
    """
    if is_sig is None:
        is_sig = torch.ones_like(beta, dtype=torch.bool)

    if is_sig.sum() == 0:
        return beta.sum() * 0.0

    unique_obj_ids = torch.unique(object_id[is_sig])
    attr_mask = object_id.view(-1, 1) == unique_obj_ids.view(1, -1)
    alphas = torch.argmax(beta.view(-1, 1) * attr_mask, dim=0)  # shape [num_objects]

    loss = torch.mean(1 - beta[alphas])
    return loss


def oc_noise_loss_per_graph(
    beta: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
):
    """
    Compute the noise penalty loss for Object Condensation per graph. Mathematically, this is the mean beta value of all noise/background points.

    Parameters
    ----------
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    is_sig : torch.Tensor
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects, by default None.

    Returns
    -------
    torch.Tensor
        Noise penalty loss.
    """
    if is_sig is None:
        is_sig = torch.ones_like(beta, dtype=torch.bool)

    is_noise = ~is_sig

    if is_noise.sum() == 0:
        return beta.sum() * 0.0

    return torch.mean(beta[is_noise])


def oc_feat_loss_per_graph(
    feat_loss: torch.Tensor,
    beta: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
):
    """
    Compute the feature loss weighted by the representative importance according to Eur. Phys. J. C 80, 886 (2020).

    Parameters
    ----------
    feat_loss : torch.Tensor
        Total feature loss of shape [num_nodes].
    beta : torch.Tensor
        Condensation strengths of shape [num_nodes].
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects, by default None.
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.

    """
    q = torch.arctanh(beta) ** 2 + q_min  # shape [num_nodes]

    if is_sig is None:
        is_sig = torch.ones_like(beta, dtype=torch.bool)

    if is_sig.sum() == 0:
        return beta.sum() * 0.0

    norm = torch.sum(q)
    sig_loss = torch.sum(feat_loss[is_sig] * q[is_sig])
    other_loss = torch.sum(feat_loss[~is_sig] * q_min)

    return (sig_loss + other_loss) / norm


def oc_attr_loss_per_graph_naive(
    x: torch.Tensor,
    beta: torch.Tensor,
    object_id: torch.Tensor,
    is_sig: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
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
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects, by default None.
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.

    Returns
    -------
    torch.Tensor
        Attractive potential loss.
    """
    if x.requires_grad:
        raise RuntimeError(
            "This is a naive implementation for testing only. Do not use in backpropagation of gradients."
        )

    if is_sig is None:
        is_sig = torch.ones_like(beta, dtype=torch.bool)

    q = torch.arctanh(beta) ** 2 + q_min  # shape [num_nodes]
    loss = x.sum() * 0.0

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
    is_sig: Optional[torch.Tensor] = None,
    q_min: float = 0.1,
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
    is_sig : Optional[torch.Tensor], optional
        Boolean mask of shape [num_nodes] indicating which nodes belong to objects. If None, all nodes are considered to belong to objects.
    q_min : float, optional
        Minimum charge value to ensure numerical stability, by default 0.1.
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

    if is_sig is None:
        is_sig = torch.ones_like(beta, dtype=torch.bool)

    beta = torch.clamp(beta, max=0.9999)
    q = torch.arctanh(beta) ** 2 + q_min  # shape [num_nodes]
    loss = x.sum() * 0.0

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
