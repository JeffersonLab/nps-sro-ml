from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import pandas as pd
import torch

from utils.graph import (
    create_unique_object_ids,
    reorder_from_graph_batches,
    pack_to_graph_batches,
)


@dataclass
class OcInferenceHyperparameters:
    beta_thres: float = 0.5
    dist_thres: float = 0.5
    pulse_score_thres: float = 0.5
    noise_idx: int = -1

    @classmethod
    def from_mapping(
        cls, hyperparameters: Mapping[str, Any] | None = None
    ) -> "OcInferenceHyperparameters":
        return cls(**(hyperparameters or {}))


from dataclasses import dataclass, field
from typing import Dict, List
import torch


@dataclass
class OcInferenceResults:
    """
    Container for inference results.

    Each entry corresponds to one event and stores the original hit-level
    inputs together with the inferred ownership information.
    """

    # original data
    event_id: List[torch.Tensor] = field(default_factory=list)
    x: List[torch.Tensor] = field(default_factory=list)
    pos: List[torch.Tensor] = field(default_factory=list)
    truth_ids: List[torch.Tensor] = field(default_factory=list)

    # inference
    min_d: List[torch.Tensor] = field(default_factory=list)
    beta: List[torch.Tensor] = field(default_factory=list)
    x_c: List[torch.Tensor] = field(default_factory=list)
    object_ids: List[torch.Tensor] = field(default_factory=list)

    def append(
        self,
        event_id: torch.Tensor,
        x: torch.Tensor,
        pos: torch.Tensor,
        truth_ids: torch.Tensor,
        min_d: torch.Tensor,
        beta: torch.Tensor,
        x_c: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> None:
        self.event_id.append(event_id)
        self.x.append(x)
        self.pos.append(pos)
        self.truth_ids.append(truth_ids)

        self.min_d.append(min_d)
        self.x_c.append(x_c)
        self.beta.append(beta)
        self.object_ids.append(object_ids)

    def to_dict(self):
        if not self.x:
            return {}

        event_id = torch.cat(
            [
                (
                    torch.as_tensor(value).reshape(-1).repeat(x.shape[0])
                    if torch.as_tensor(value).numel() == 1
                    else torch.as_tensor(value).reshape(-1)
                )
                for value, x in zip(self.event_id, self.x)
            ]
        )
        x = torch.cat(self.x, dim=0)
        x_c = torch.cat(self.x_c, dim=0)
        pos = torch.cat(self.pos, dim=0)

        return {
            "event_id": event_id.detach().cpu().numpy(),
            "truth_ids": torch.cat(self.truth_ids).detach().cpu().flatten().numpy(),
            "min_d": torch.cat(self.min_d).detach().cpu().flatten().numpy(),
            "beta": torch.cat(self.beta).detach().cpu().flatten().numpy(),
            "object_ids": torch.cat(self.object_ids).detach().cpu().flatten().numpy(),
            **{
                f"x_{idx}": x[:, idx].detach().cpu().numpy()
                for idx in range(x.shape[1])
            },
            **{
                f"x_c_{idx}": x_c[:, idx].detach().cpu().numpy()
                for idx in range(x_c.shape[1])
            },
            **{
                f"pos_{idx}": pos[:, idx].detach().cpu().numpy()
                for idx in range(pos.shape[1])
            },
        }

    def to_df(self):
        return pd.DataFrame(self.to_dict())

    def to_csv(self, filename):
        self.to_df().to_csv(filename, index=False)


class ObjectCondensationInferencer:

    def __init__(
        self,
        model: torch.nn.Module,
        hyperparameters: OcInferenceHyperparameters | None = None,
        config: Optional[dict] = None,
    ):
        self.model = model
        self.hyperparameters = hyperparameters or OcInferenceHyperparameters()
        self.config = config or {}

    def _infer(self, data: Any, results: OcInferenceResults, event_id: int) -> int:

        data = self._preprocess(data)
        data = data.to(self.device)

        x = data.x
        y = data.y.squeeze(-1).long()
        pos = data.pos
        batch = (
            data.batch
            if hasattr(data, "batch")
            else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        )

        noise_idx = self.hyperparameters.get("noise_idx", -1)
        truth_ids = create_unique_object_ids(y, batch, noise_idx)

        outs, idx_out, node_mask = pack_to_graph_batches(x, [pos], batch=batch)
        x, pos = outs[0], outs[1]
        x_c, beta = self.model(x, pos, node_mask)

        x_c = reorder_from_graph_batches(x_c, idx_out)
        beta = reorder_from_graph_batches(beta, idx_out)
        beta = beta.squeeze(-1)

        for b in batch.unique(sorted=True):
            b_mask = batch == b
            cluster_ids, min_d = oc_inference_per_graph(
                x_c[b_mask],
                beta[b_mask],
                beta_thres=self.hyperparameters.beta_thres,
                dist_thres=self.hyperparameters.dist_thres,
                bkg_idx=self.hyperparameters.noise_idx,
            )
            results.append(
                event_id=event_id,
                x=x[b_mask],
                pos=pos[b_mask],
                truth_ids=truth_ids[b_mask],
                min_d=min_d,
                beta=beta[b_mask],
                x_c=x_c[b_mask],
                cluster_ids=cluster_ids,
            )
            event_id += 1

        return event_id

    def infer(self, dataloader: Any) -> OcInferenceResults:
        results = OcInferenceResults()
        event_id = 0

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.model.device)
                event_id = self._infer(data, results, event_id)

        return results


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
            torch.full(
                (x.size(0),), fill_value=bkg_idx, dtype=torch.long, device=x.device
            ),
            torch.full(
                (x.size(0),), fill_value=float('inf'), dtype=x.dtype, device=x.device
            ),
        )

    d = torch.cdist(x, x[seed_mask], p=2)  #   [N, S]

    # assign object ID to all nodes based on closest seed
    min_d, obj_ids = torch.min(d, dim=1)
    obj_ids = obj_ids + bkg_idx + 1  # [N]
    # replulsion based on distance threshold
    obj_ids[min_d > dist_thres] = bkg_idx  # background

    return obj_ids, min_d
