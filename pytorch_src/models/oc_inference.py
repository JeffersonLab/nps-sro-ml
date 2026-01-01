import torch


def oc_inference_per_batch(
    x: torch.Tensor,
    beta: torch.Tensor,
    batch: torch.Tensor,
    beta_thres: float = 0.4,
    dist_thres: float = 0.8,
    bkg_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform clustering in latent space based on beta and distance thresholding for a batch of graphs.

    Parameters
    ----------
    x : torch.Tensor
        Latent space positions, shape [num_nodes, d_model]
    beta : torch.Tensor
        Condensation strengths, shape [num_nodes] or [num_nodes, 1]
    batch : torch.Tensor
        Batch vector indicating graph membership of each node, shape [num_nodes]
    beta_thres : float, optional
        Minimum beta to be considered for clustering, by default 0.4
    dist_thres : float, optional
        Maximum distance to cluster points together, by default 0.8
    bkg_idx : int, optional
        Index to use for background/unassigned, by default 0

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple of tensors in the original node order:
        - cluster_ids: Tensor of shape [num_nodes] with cluster assignments
        - min_d: Tensor of shape [num_nodes] with minimum distance to assigned cluster center
    """
    beta_ = beta.view(-1)

    seed_mask = (beta_ > beta_thres).nonzero(as_tuple=True)[0]  # [S]

    obj_ids = torch.full(
        (x.size(0),),
        fill_value=bkg_idx,
        dtype=torch.long,
        device=x.device,
    )
    min_d = torch.full(
        (x.size(0),),
        fill_value=float('inf'),
        dtype=x.dtype,
        device=x.device,
    )

    if len(seed_mask) == 0:
        return obj_ids, min_d

    batch_mask = batch[:, None] == batch[seed_mask][None, :]

    d = torch.cdist(x, x[seed_mask], p=2)  #   [N, S]
    d[~batch_mask] = float('inf')
    has_seed = batch_mask.any(dim=1)

    # assign object ID to all nodes based on closest seed
    min_d[has_seed], obj_ids[has_seed] = d[has_seed].min(dim=1)

    obj_ids[has_seed] += bkg_idx + 1  # [N]

    # replulsion based on distance threshold
    obj_ids[min_d > dist_thres] = bkg_idx  # background

    return obj_ids, min_d


def oc_inference_per_graph(
    x: torch.Tensor,
    beta: torch.Tensor,
    beta_thres: float = 0.4,
    dist_thres: float = 0.8,
    bkg_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform clustering in latent space based on beta and distance thresholding.

    Parameters
    ----------
    x : torch.Tensor
        Latent space positions, shape [num_nodes, d_model]
    beta : torch.Tensor
        Condensation strengths, shape [num_nodes] or [num_nodes, 1]
    beta_thres : float, optional
        Minimum beta to be considered for clustering, by default 0.4
    dist_thres : float, optional
        Maximum distance to cluster points together, by default 0.8
    bkg_idx : int, optional
        Index to use for background/unassigned, by default 0

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        cluster_ids: Tensor of shape [num_nodes] with cluster assignments
        min_d: Tensor of shape [num_nodes] with minimum distance to assigned cluster center
    """
    beta_ = beta.view(-1)
    seed_mask = (beta_ > beta_thres).nonzero(as_tuple=True)[0]  # [S]

    if len(seed_mask) == 0:
        return (
            torch.full_like(beta, fill_value=bkg_idx, dtype=torch.long),
            torch.full_like(beta, fill_value=float('inf'), dtype=beta.dtype),
        )

    d = torch.cdist(x, x[seed_mask], p=2)  #   [N, S]

    # assign object ID to all nodes based on closest seed
    min_d, obj_ids = torch.min(d, dim=1)
    obj_ids = obj_ids + bkg_idx + 1  # [N]
    # replulsion based on distance threshold
    obj_ids[min_d > dist_thres] = bkg_idx  # background

    return obj_ids, min_d
