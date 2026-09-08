from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import torch

from .oc_inference import (
    BaseOcInferenceHyperparameters,
    BaseOcInferenceManager,
    BaseOcInferenceResults,
    BaseOcInferenceResultsPerGraph,
    oc_inference_per_graph,
)


@dataclass
class VtpHitOcInferenceHyperparameters(BaseOcInferenceHyperparameters):
    """Hyperparameters for VTP hit-level OC inference."""

    sig_thres: float = 0.5

    q_min: float = 0.3


@dataclass
class VtpHitOcInferenceResultsPerGraph(BaseOcInferenceResultsPerGraph):
    """Hit-level OC results for one event."""

    x_signal: torch.Tensor
    is_triggered: torch.Tensor


class VtpHitOcInferenceResults(BaseOcInferenceResults):
    """Hit-level OC results for multiple events."""

    result_type: ClassVar[type[VtpHitOcInferenceResultsPerGraph]] = (
        VtpHitOcInferenceResultsPerGraph
    )


class VtpHitOcInferenceManager(BaseOcInferenceManager):
    """Run hit-level OC inference, including signal classification."""

    hyperparameters_type: ClassVar[type[VtpHitOcInferenceHyperparameters]] = (
        VtpHitOcInferenceHyperparameters
    )
    results_type: ClassVar[type[VtpHitOcInferenceResults]] = VtpHitOcInferenceResults

    def __init__(
        self,
        model: torch.nn.Module,
        hyperparameters: (
            VtpHitOcInferenceHyperparameters | Mapping[str, Any] | None
        ) = None,
    ):
        super().__init__(model, hyperparameters)

    def _prepare_model_inputs(self, data: Any) -> tuple[torch.Tensor, ...]:
        """Scale raw energy, time, and detector coordinates for the hit model."""
        num_columns = 30
        num_rows = 36
        num_time_bins = 110

        energy = data.x[:, 0]
        scaled_time = 2 * data.x[:, 1] / num_time_bins - 1
        scaled_energy = energy / 1600
        log_energy = torch.log1p(energy)

        scaled_x = 2 * data.pos[:, 0] / num_columns - 1
        scaled_y = 2 * data.pos[:, 1] / num_rows - 1

        x = torch.stack([scaled_energy, log_energy, scaled_time], dim=-1)
        pos = torch.stack([scaled_x, scaled_y], dim=-1)
        return x, pos

    def _infer_graph(
        self,
        *model_outputs: tuple[torch.Tensor, ...],
    ) -> Mapping[str, Any]:

        x_c, beta, x_signal = model_outputs
        x_signal = x_signal.squeeze(-1) if x_signal.ndim > 1 else x_signal

        object_ids, min_d = oc_inference_per_graph(
            x_c,
            beta,
            beta_thres=self.hyperparameters.beta_thres,
            dist_thres=self.hyperparameters.dist_thres,
            empty_idx=self.hyperparameters.empty_idx,
        )

        q = torch.arctanh(beta) ** 2 + self.hyperparameters.q_min

        # aggregate logits for hits belonging to the same object
        object_ids_unique = object_ids.unique(sorted=True)
        object_ids_unique = object_ids_unique[
            object_ids_unique != self.hyperparameters.empty_idx
        ]
        for obj_id in object_ids_unique:
            obj_mask = object_ids == obj_id
            q_obj = q[obj_mask]
            x_signal_obj = x_signal[obj_mask]
            x_signal[obj_mask] = torch.sum(q_obj * x_signal_obj) / torch.sum(q_obj)

        return {
            "object_ids": object_ids,
            "x_c": x_c,
            "beta": beta,
            "min_d": min_d,
            "x_signal": x_signal,
            "is_triggered": torch.sigmoid(x_signal) > self.hyperparameters.sig_thres,
        }
