import torch
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import pair_confusion_matrix

from .oc_inference import (
    BaseOcInferenceHyperparameters,
    BaseOcInferenceManager,
    BaseOcInferenceResults,
    BaseOcInferenceResultsPerGraph,
    oc_inference_per_graph,
)

from .report_helper import plot_confusion_matrix, plot_event_objects
from .metrics import metrics_from_confusion, prediction_scores, clustering_scores

from utils.utils import write_json
from utils.graph import create_unique_object_ids


@dataclass
class VtpHitOcInferenceHyperparameters(BaseOcInferenceHyperparameters):
    """Hyperparameters for VTP hit-level OC inference."""

    sig_thres: float = 0.5
    q_min: float = 0.3


@dataclass
class VtpHitOcInferenceResultsPerGraph(BaseOcInferenceResultsPerGraph):
    """Hit-level OC results for one event."""

    cluster_type: (
        torch.Tensor
    )  # truth cluster type for each hit [0 = not triggered, 1 = triggered]
    trigger_logit: torch.Tensor  # model output logits for trigger classification
    trigger_probability: (
        torch.Tensor
    )  # model output probabilities for trigger classification
    is_triggered: torch.Tensor  # binary prediction for trigger classification


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

    def _extract_input_data(self, data: Any) -> Mapping[str, Any]:

        y = data.y.squeeze(-1).long()
        batch = (
            data.batch
            if hasattr(data, "batch")
            else torch.zeros(y.shape[0], dtype=torch.long, device=y.device)
        )

        empty_idx = self.hyperparameters.empty_idx
        truth_ids = create_unique_object_ids(y, batch, empty_idx)
        return {
            "truth_ids": truth_ids,
            "x": data.x,
            "pos": data.pos,
            "batch": batch,
            "cluster_type": data.cluster_type,
        }

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

        beta_ = beta.clamp(
            min=0.0,
            max=1.0 - torch.finfo(beta.dtype).eps,
        )
        q = torch.arctanh(beta_) ** 2 + self.hyperparameters.q_min

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

        trigger_prob = torch.sigmoid(x_signal)
        return {
            "object_ids": object_ids,
            "x_c": x_c,
            "beta": beta,
            "min_d": min_d,
            "trigger_logit": x_signal,
            "trigger_probability": trigger_prob,
            "is_triggered": trigger_prob > self.hyperparameters.sig_thres,
        }

    def report(self, save_dir: pathlib.Path, **kwargs) -> None:
        """
        Export inference results to the specified directory and generate performance artifacts.

        Parameters
        ----------
        save_dir : pathlib.Path
            Directory to save the results and reports.

        kwargs : dict
            Additional keyword arguments for report generation.

        """
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        df_results = self.results.to_df()
        if df_results.empty:
            raise ValueError("No inference results available.")

        self.export(save_dir / "results.csv", index=False)
        self.export(save_dir / "results.json")

        trig_cm = np.zeros((2, 2), dtype=int)
        hit_cm = np.zeros((2, 2), dtype=int)
        clustering_metrics = []

        if (df_results["truth_ids"] == self.hyperparameters.empty_idx).any():
            raise RuntimeError("Empty truth IDs found in the results.")

        unique_events = df_results["event_id"].unique()

        for i in unique_events:
            df_event = df_results[df_results["event_id"] == i]

            trig_cm += confusion_matrix(
                df_event["cluster_type"],
                df_event["is_triggered"],
                labels=[0, 1],
            )

            num_unassigned = (
                df_event["object_ids"] == self.hyperparameters.empty_idx
            ).sum()

            df_event.loc[
                df_event["object_ids"] == self.hyperparameters.empty_idx, "object_ids"
            ] = range(
                df_event["object_ids"].max() + 1,
                df_event["object_ids"].max() + 1 + num_unassigned,
            )

            truth_ids = df_event["truth_ids"]
            object_ids = df_event["object_ids"]

            hit_cm += pair_confusion_matrix(truth_ids, object_ids)

            clustering_metrics.append(clustering_scores(truth_ids, object_ids))

        hit_metrics = metrics_from_confusion(hit_cm)

        clus_metrics = (
            {
                name: {
                    "mean": float(np.mean([m[name] for m in clustering_metrics])),
                    "std": float(np.std([m[name] for m in clustering_metrics])),
                }
                for name in clustering_metrics[0].keys()
            }
            if clustering_metrics
            else {}
        )

        trig_metrics = prediction_scores(
            truth=df_results["cluster_type"],
            prediction=df_results["is_triggered"],
            prob=df_results["trigger_probability"],
        )

        write_json(
            {
                "num_events": int(df_results["event_id"].nunique()),
                "num_hits": int(len(df_results)),
                "hit_aggregation": {
                    **hit_metrics,
                    "event_metrics": clus_metrics,
                },
                "trigger": trig_metrics,
                "confusion_matrices": {
                    "hit_aggregation": hit_cm,
                    "trigger": trig_cm,
                },
            },
            save_dir / "metrics.json",
        )

        fig_dir = save_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        plot_confusion_matrix(
            trig_cm,
            row_labels=[r"$\mathrm{Not\ Triggered}$", r"$\mathrm{Triggered}$"],
            column_labels=[r"$\mathrm{Not\ Triggered}$", r"$\mathrm{Triggered}$"],
            title=r"$\mathrm{Trigger\ Confusion\ Matrix}$",
            output_path=fig_dir / "trigger_confusion_matrix.png",
        )

        plot_confusion_matrix(
            hit_cm,
            row_labels=[
                r"$\mathrm{Truth\ Different\ Object}$",
                r"$\mathrm{Truth\ Same\ Object}$",
            ],
            column_labels=[
                r"$\mathrm{Predicted\ Different\ Object}$",
                r"$\mathrm{Predicted\ Same\ Object}$",
            ],
            title=r"$\mathrm{Hit\ Aggregation\ Confusion\ Matrix}$",
            output_path=fig_dir / "hit_confusion_matrix.png",
        )

        seed = kwargs.pop("seed", 42)
        nplots = kwargs.pop("num_det_plots", 10)

        rng = np.random.default_rng(seed)

        for i in rng.choice(
            unique_events, size=min(nplots, len(unique_events)), replace=False
        ):
            df_event = df_results[df_results["event_id"] == i]
            plot_event_objects(
                truth_ids=df_event["truth_ids"].tolist(),
                pred_ids=df_event["object_ids"].tolist(),
                pos=np.column_stack(
                    [
                        df_event["pos_0"].to_numpy(),
                        df_event["pos_1"].to_numpy(),
                    ]
                ),
                empty_idx=self.hyperparameters.empty_idx,
                output_path=fig_dir / f"event_{i}_objects.png",
            )
