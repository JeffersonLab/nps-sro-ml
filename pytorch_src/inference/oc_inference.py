from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

import pandas as pd
import torch

from utils.graph import (
    create_unique_object_ids,
    reorder_from_graph_batches,
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


@dataclass
class OcInferenceResults:
    event_id: list[torch.Tensor] = field(default_factory=list)
    cluster_ids: list[torch.Tensor] = field(default_factory=list)
    min_d: list[torch.Tensor] = field(default_factory=list)
    beta: list[torch.Tensor] = field(default_factory=list)
    object_ids: list[torch.Tensor] = field(default_factory=list)
    det_x: list[torch.Tensor] = field(default_factory=list)
    det_y: list[torch.Tensor] = field(default_factory=list)
    x_c: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    pulse_fields: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def append_graph(
        self,
        *,
        event_id: int,
        cluster_ids: torch.Tensor,
        min_d: torch.Tensor,
        beta: torch.Tensor,
        x_c: torch.Tensor,
        object_ids: torch.Tensor,
        pos: torch.Tensor,
        pulse_cluster_ids: Optional[torch.Tensor] = None,
        pulse_min_d: Optional[torch.Tensor] = None,
        pulse_beta: Optional[torch.Tensor] = None,
        pulse_score: Optional[torch.Tensor] = None,
        pulse_object_ids: Optional[torch.Tensor] = None,
        pulse_x_c: Optional[torch.Tensor] = None,
    ) -> None:
        nb = cluster_ids.numel()
        self.event_id.append(torch.full((nb,), event_id, dtype=torch.long).cpu())
        self.cluster_ids.append(cluster_ids.view(-1).cpu())
        self.min_d.append(min_d.view(-1).cpu())
        self.beta.append(beta.view(-1).cpu())
        self.object_ids.append(object_ids.view(-1).cpu())
        self.det_x.append(pos[:, 0].view(-1).cpu())
        self.det_y.append(pos[:, 1].view(-1).cpu())

        for dim in range(x_c.size(1)):
            key = f"x_c_{dim}"
            self.x_c.setdefault(key, []).append(x_c[:, dim].view(-1).cpu())

        if pulse_cluster_ids is not None:
            self._append_slot_tensor(self.pulse_fields, "pulse_cluster_ids", pulse_cluster_ids)
        if pulse_min_d is not None:
            self._append_slot_tensor(self.pulse_fields, "pulse_min_d", pulse_min_d)
        if pulse_beta is not None:
            self._append_slot_tensor(self.pulse_fields, "pulse_beta", pulse_beta)
        if pulse_score is not None:
            self._append_slot_tensor(self.pulse_fields, "pulse_score", pulse_score)
        if pulse_object_ids is not None:
            self._append_slot_tensor(self.pulse_fields, "pulse_object_ids", pulse_object_ids)
        if pulse_x_c is not None:
            self._append_slot_tensor(self.pulse_fields, "pulse_x_c", pulse_x_c)

    def _append_slot_tensor(
        self,
        store: dict[str, list[torch.Tensor]],
        prefix: str,
        tensor: torch.Tensor,
    ) -> None:
        if tensor.ndim == 2:
            for slot in range(tensor.size(1)):
                key = f"{prefix}_{slot}"
                store.setdefault(key, []).append(tensor[:, slot].view(-1).cpu())
            return

        if tensor.ndim == 3:
            for slot in range(tensor.size(1)):
                for dim in range(tensor.size(2)):
                    key = f"{prefix}_{slot}_{dim}"
                    store.setdefault(key, []).append(tensor[:, slot, dim].view(-1).cpu())
            return

        raise ValueError(
            f"Expected slot tensor with ndim 2 or 3, got shape {tuple(tensor.shape)}."
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self.event_id:
            columns = [
                "event_id",
                "cluster_ids",
                "min_d",
                "beta",
                "object_ids",
                "det_x",
                "det_y",
            ]
            return pd.DataFrame(columns=columns)

        result_tensors = {}
        for key, values in asdict(self).items():
            if key in {"x_c", "pulse_fields"}:
                continue
            result_tensors[key] = torch.cat(values, dim=0).numpy()
        result_tensors.update(
            {key: torch.cat(values, dim=0).numpy() for key, values in self.x_c.items()}
        )
        result_tensors.update(
            {
                key: torch.cat(values, dim=0).numpy()
                for key, values in self.pulse_fields.items()
            }
        )
        return pd.DataFrame(result_tensors)


def normalize_y_object_ids(
    y: torch.Tensor,
    batch: torch.Tensor,
    noise_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    elif y.ndim == 2 and y.shape[-1] == 1:
        pass
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)

    y = y.long()
    token_object_ids = torch.full_like(y, noise_idx)
    for slot in range(y.shape[1]):
        token_object_ids[:, slot] = create_unique_object_ids(
            y[:, slot], batch, noise_idx=noise_idx
        )

    node_object_ids = torch.full(
        (y.shape[0],),
        noise_idx,
        dtype=torch.long,
        device=y.device,
    )
    signal_mask = token_object_ids != noise_idx
    has_signal = signal_mask.any(dim=-1)
    first_signal_idx = signal_mask.long().argmax(dim=-1)
    node_object_ids[has_signal] = token_object_ids[
        has_signal, first_signal_idx[has_signal]
    ]
    return node_object_ids, token_object_ids


class BaseObjectCondensationInferencer:
    """
    Base OC inferencer that prepares model inputs in the same packed format used
    by the OC trainer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        hyperparameters: OcInferenceHyperparameters | None = None,
        config: Optional[dict] = None,
    ):
        self.model = model
        self.hyperparameters = hyperparameters or OcInferenceHyperparameters()
        self.config = config or {}
        self.model.configure_input_preprocessing(self.config)

    def _predict(
        self, data: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.model.preprocess_features(data)
        pos = data.pos
        batch = self.model.get_batch_vector(x, getattr(data, "batch", None))

        x_graph, pos_graph, fea_mask, node_mask, idx_out = self.model.prepare_graph_inputs(
            x, pos, batch
        )
        x_c, beta = self.model(x_graph, pos_graph, fea_mask, node_mask)
        x_c = reorder_from_graph_batches(x_c, idx_out)
        beta = reorder_from_graph_batches(beta, idx_out).squeeze(-1)
        return x_c, beta, pos, batch

    def infer_data(
        self,
        data: Any,
        results: OcInferenceResults,
        event_id: int,
    ) -> int:
        y = data.y.squeeze(-1).long()
        x_c, beta, pos, batch = self._predict(data)
        object_ids = create_unique_object_ids(
            y,
            batch,
            noise_idx=self.hyperparameters.noise_idx,
        )

        for b in batch.unique(sorted=True):
            b_mask = batch == b
            cluster_ids, min_d = oc_inference_per_graph(
                x_c[b_mask],
                beta[b_mask],
                beta_thres=self.hyperparameters.beta_thres,
                dist_thres=self.hyperparameters.dist_thres,
                bkg_idx=self.hyperparameters.noise_idx,
            )
            results.append_graph(
                event_id=event_id,
                cluster_ids=cluster_ids,
                min_d=min_d,
                beta=beta[b_mask],
                x_c=x_c[b_mask],
                object_ids=object_ids[b_mask],
                pos=pos[b_mask],
            )
            event_id += 1

        return event_id

    def infer_dataloader(self, dataloader: Any) -> OcInferenceResults:
        results = OcInferenceResults()
        event_id = 0

        with torch.no_grad():
            for data in dataloader:
                data = data.to(next(self.model.parameters()).device)
                event_id = self.infer_data(data, results, event_id)

        return results


class PulseOCInferencer(BaseObjectCondensationInferencer):
    pass


class WaveformOCInferencer(BaseObjectCondensationInferencer):
    pass


class MultiPulseOCInferencer(BaseObjectCondensationInferencer):
    def _predict(
        self,
        data: Any,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        x = self.model.preprocess_features(data)
        pos = data.pos
        batch = self.model.get_batch_vector(x, getattr(data, "batch", None))

        x_graph, pos_graph, fea_mask, node_mask, idx_out = self.model.prepare_graph_inputs(
            x, pos, batch
        )
        outputs = self.model(x_graph, pos_graph, fea_mask, node_mask)
        outputs = {
            key: reorder_from_graph_batches(value, idx_out)
            for key, value in outputs.items()
        }
        return outputs, pos, batch

    def _pool_node_outputs(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pulse_score = outputs["pulse_score"]
        pulse_beta = outputs["pulse_beta"]
        pulse_x_c = outputs["pulse_x_c"]
        token_mask = outputs["token_mask"]

        token_weight = pulse_score * token_mask.to(pulse_score.dtype)
        token_weight = token_weight / token_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        x_c = (pulse_x_c * token_weight.unsqueeze(-1)).sum(dim=2)
        beta = pulse_beta.max(dim=-1).values
        return x_c, beta

    def _predict_multi_object_ids(
        self,
        pulse_x_c: torch.Tensor,
        pulse_beta: torch.Tensor,
        pulse_score: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = pulse_beta.device
        noise_idx = self.hyperparameters.noise_idx
        pulse_cluster_ids = torch.full(
            pulse_beta.shape,
            fill_value=noise_idx,
            dtype=torch.long,
            device=device,
        )
        pulse_min_d = torch.full(
            pulse_beta.shape,
            fill_value=float("inf"),
            dtype=pulse_x_c.dtype,
            device=device,
        )

        active_mask = token_mask & (pulse_score > self.hyperparameters.pulse_score_thres)
        if active_mask.sum() == 0:
            return pulse_cluster_ids, pulse_min_d

        cluster_ids, min_d = oc_inference_per_graph(
            pulse_x_c[active_mask],
            pulse_beta[active_mask],
            beta_thres=self.hyperparameters.beta_thres,
            dist_thres=self.hyperparameters.dist_thres,
            bkg_idx=noise_idx,
        )
        pulse_cluster_ids[active_mask] = cluster_ids
        pulse_min_d[active_mask] = min_d
        return pulse_cluster_ids, pulse_min_d

    def infer_data(
        self,
        data: Any,
        results: OcInferenceResults,
        event_id: int,
    ) -> int:
        outputs, pos, batch = self._predict(data)
        x_c, beta = self._pool_node_outputs(outputs)
        object_ids, token_object_ids = normalize_y_object_ids(
            data.y,
            batch,
            noise_idx=self.hyperparameters.noise_idx,
        )

        for b in batch.unique(sorted=True):
            b_mask = batch == b
            cluster_ids, min_d = oc_inference_per_graph(
                x_c[b_mask],
                beta[b_mask],
                beta_thres=self.hyperparameters.beta_thres,
                dist_thres=self.hyperparameters.dist_thres,
                bkg_idx=self.hyperparameters.noise_idx,
            )
            pulse_cluster_ids, pulse_min_d = self._predict_multi_object_ids(
                outputs["pulse_x_c"][b_mask],
                outputs["pulse_beta"][b_mask],
                outputs["pulse_score"][b_mask],
                outputs["token_mask"][b_mask],
            )
            results.append_graph(
                event_id=event_id,
                cluster_ids=cluster_ids,
                min_d=min_d,
                beta=beta[b_mask],
                x_c=x_c[b_mask],
                object_ids=object_ids[b_mask],
                pos=pos[b_mask],
                pulse_cluster_ids=pulse_cluster_ids,
                pulse_min_d=pulse_min_d,
                pulse_beta=outputs["pulse_beta"][b_mask],
                pulse_score=outputs["pulse_score"][b_mask],
                pulse_object_ids=token_object_ids[b_mask],
                pulse_x_c=outputs["pulse_x_c"][b_mask],
            )
            event_id += 1

        return event_id


def build_oc_inferencer(
    model: torch.nn.Module,
    config: Optional[dict] = None,
    hyperparameters: OcInferenceHyperparameters | None = None,
) -> BaseObjectCondensationInferencer:
    config = config or {}
    trainer_type = config.get("trainer", {}).get("type")
    input_type = getattr(model, "input_type", None)

    if trainer_type == "MultiPulseOCTrainer":
        return MultiPulseOCInferencer(model, hyperparameters=hyperparameters, config=config)

    if trainer_type == "WaveformOCTrainer" or input_type == "waveform":
        return WaveformOCInferencer(model, hyperparameters=hyperparameters, config=config)

    if trainer_type == "PulseOCTrainer" or input_type == "pulse_set":
        return PulseOCInferencer(model, hyperparameters=hyperparameters, config=config)

    return BaseObjectCondensationInferencer(
        model,
        hyperparameters=hyperparameters,
        config=config,
    )


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
            torch.full((x.size(0),), fill_value=bkg_idx, dtype=torch.long, device=x.device),
            torch.full((x.size(0),), fill_value=float('inf'), dtype=x.dtype, device=x.device),
        )

    d = torch.cdist(x, x[seed_mask], p=2)  #   [N, S]

    # assign object ID to all nodes based on closest seed
    min_d, obj_ids = torch.min(d, dim=1)
    obj_ids = obj_ids + bkg_idx + 1  # [N]
    # replulsion based on distance threshold
    obj_ids[min_d > dist_thres] = bkg_idx  # background

    return obj_ids, min_d
