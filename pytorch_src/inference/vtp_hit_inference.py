import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import pair_confusion_matrix

from .oc_inference import (
    BaseOcInferenceHyperparameters,
    BaseOcInferenceManager,
    BaseOcInferenceResults,
    BaseOcInferenceResultsPerGraph,
    oc_inference_per_graph,
)

from .metrics import metrics_from_confusion, prediction_scores, clustering_scores

from utils.utils import write_json
from utils.graph import create_unique_object_ids

from .report_helper import (
    get_obj_stats,
    plot_distributions,
    plot_beta_distribution,
    plot_min_distance_distribution,
    plot_confusion_matrix,
)


@dataclass
class VtpHitOcInferenceHyperparameters(BaseOcInferenceHyperparameters):
    """Hyperparameters for VTP hit-level OC inference."""

    sig_thres: float = 0.5
    q_min: float = 0.3


@dataclass
class VtpHitOcInferenceResultsPerGraph(BaseOcInferenceResultsPerGraph):
    """Hit-level OC results for one event."""

    energy: torch.Tensor  # calibrated en, mev
    time: torch.Tensor  # timestamp, not ns
    col: torch.Tensor
    row: torch.Tensor
    cluster_type: torch.Tensor  # [0 = not triggered, 1 = triggered]
    truth_ids: torch.Tensor  # unique object IDs for the ground truth hits

    # model output logits for trigger classification
    trigger_logit: torch.Tensor

    # model output probabilities for trigger classification
    trigger_probability: torch.Tensor
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

    def _define_batch(self, data: Any) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            return torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)
        return batch

    def _extract_truth(
        self,
        data: Any,
        *,
        batch: torch.Tensor,
    ) -> Mapping[str, Any]:

        y = data.y.squeeze(-1).long()
        empty_idx = self.hyperparameters.empty_idx
        truth_ids = create_unique_object_ids(y, batch, empty_idx)
        return {
            "energy": data.x[:, 0],
            "time": data.x[:, 1],
            "col": data.pos[:, 0],
            "row": data.pos[:, 1],
            "cluster_type": data.cluster_type,
            "truth_ids": truth_ids,
        }

    def _prepare_model_inputs(
        self,
        data: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Scale raw energy, time, and detector coordinates for the hit model."""
        from datasets.nps import NCOLS, NROWS, NTIME

        energy = data.x[:, 0]
        scaled_time = 2 * data.x[:, 1] / NTIME - 1
        scaled_energy = energy / 1600
        log_energy = torch.log1p(energy)

        scaled_x = 2 * data.pos[:, 0] / NCOLS - 1
        scaled_y = 2 * data.pos[:, 1] / NROWS - 1

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
        fig_dir = save_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        results = self.results.to_dict()
        self.export(save_dir / "results.csv", index=False)
        self.export(save_dir / "results.json")

        stats_summary: dict[str, Counter] = collect_stats_summary(
            results, bkg_ids=[self.hyperparameters.empty_idx]
        )
        metrics = get_metrics(results, bkg_ids=[self.hyperparameters.empty_idx])

        write_json(stats_summary, save_dir / "stats_summary.json")
        write_json(metrics, save_dir / "metrics.json")

        plot_confusion_matrix(
            metrics["trig_cm"],
            row_labels=[r"$\mathrm{Not\ Triggered}$", r"$\mathrm{Triggered}$"],
            column_labels=[r"$\mathrm{Not\ Triggered}$", r"$\mathrm{Triggered}$"],
            title=r"$\mathrm{Trigger\ Confusion\ Matrix}$",
            output_path=fig_dir / "trigger_confusion_matrix.png",
        )

        plot_confusion_matrix(
            metrics["aggr_cm"],
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

        plot_confusion_matrix(
            metrics["trig_aggr_cm"],
            row_labels=[
                r"$\mathrm{Truth\ Different\ Object}$",
                r"$\mathrm{Truth\ Same\ Object}$",
            ],
            column_labels=[
                r"$\mathrm{Predicted\ Different\ Object}$",
                r"$\mathrm{Predicted\ Same\ Object}$",
            ],
            title=r"$\mathrm{Triggered\ Hit\ Aggregation\ Confusion\ Matrix}$",
            output_path=fig_dir / "triggered_hit_confusion_matrix.png",
        )

        plot_beta_distribution(
            beta=results["beta"],
            output_path=fig_dir / "beta_distribution.png",
        )
        plot_min_distance_distribution(
            min_d=results["min_d"],
            output_path=fig_dir / "min_distance_distribution.png",
        )

        plot_obj_size_distribution(
            true_trig_obj_size=stats_summary["true_trig"]["obj_sizes"],
            true_non_trig_obj_size=stats_summary["true_non_trig"]["obj_sizes"],
            pred_trig_obj_size=stats_summary["pred_trig"]["obj_sizes"],
            pred_non_trig_obj_size=stats_summary["pred_non_trig"]["obj_sizes"],
            output_path=fig_dir / "object_size_distribution.png",
        )

        plot_num_objects_distribution(
            true_trig_nobjs=stats_summary["true_trig"]["nobjs"],
            true_non_trig_nobjs=stats_summary["true_non_trig"]["nobjs"],
            pred_trig_nobjs=stats_summary["pred_trig"]["nobjs"],
            pred_non_trig_nobjs=stats_summary["pred_non_trig"]["nobjs"],
            output_path=fig_dir / "num_objects_distribution.png",
        )

        generate_event_object_plots(
            event_ids=results["event_id"],
            truth_ids=results["truth_ids"],
            pred_ids=results["object_ids"],
            pos=np.column_stack(
                [
                    results["col"],
                    results["row"],
                ]
            ),
            bkg_ids=self.hyperparameters.empty_idx,
            fig_dir=fig_dir / "event_objects",
            seed=kwargs.get("seed", 42),
            nplots=kwargs.get("num_det_plots", 10),
        )


def plot_obj_size_distribution(
    true_trig_obj_size: dict[int, int],
    true_non_trig_obj_size: dict[int, int],
    pred_trig_obj_size: dict[int, int],
    pred_non_trig_obj_size: dict[int, int],
    output_path: Optional[pathlib.Path] = None,
) -> None:

    true_trig_sizes = list(true_trig_obj_size.keys())
    true_trig_counts = list(true_trig_obj_size.values())
    true_non_trig_sizes = list(true_non_trig_obj_size.keys())
    true_non_trig_counts = list(true_non_trig_obj_size.values())
    pred_trig_sizes = list(pred_trig_obj_size.keys())
    pred_trig_counts = list(pred_trig_obj_size.values())
    pred_non_trig_sizes = list(pred_non_trig_obj_size.keys())
    pred_non_trig_counts = list(pred_non_trig_obj_size.values())

    max_size = max(
        set(
            true_trig_sizes
            + true_non_trig_sizes
            + pred_trig_sizes
            + pred_non_trig_sizes
        )
    )

    fig, ax = plot_distributions(
        arrs=[
            true_trig_sizes,
            true_non_trig_sizes,
            pred_trig_sizes,
            pred_non_trig_sizes,
        ],
        bins=max_size,
        range=(0, max_size),
        weights=[
            true_trig_counts,
            true_non_trig_counts,
            pred_trig_counts,
            pred_non_trig_counts,
        ],
        hist_kwargs=[
            {
                "histtype": "stepfilled",
                "color": "blue",
                "alpha": 0.7,
                "label": r"$\mathrm{true\ trig}$",
            },
            {
                "histtype": "stepfilled",
                "color": "orange",
                "alpha": 0.7,
                "label": r"$\mathrm{true\ non-trig}$",
            },
            {
                "histtype": "stepfilled",
                "color": "green",
                "alpha": 0.7,
                "label": r"$\mathrm{pred\ trig}$",
            },
            {
                "histtype": "stepfilled",
                "color": "red",
                "alpha": 0.7,
                "label": r"$\mathrm{pred\ non-trig}$",
            },
        ],
        fig_kwargs={"figsize": (5, 4), "constrained_layout": True, "dpi": 300},
    )
    ax.set_xlabel(r"$\mathrm{Object\ Size}$")
    ax.set_ylabel(r"$\mathrm{Counts}$")
    ax.legend(
        loc="best",
        fontsize=10,
        frameon=False,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_num_objects_distribution(
    true_trig_nobjs: dict[int, int],
    true_non_trig_nobjs: dict[int, int],
    pred_trig_nobjs: dict[int, int],
    pred_non_trig_nobjs: dict[int, int],
    output_path: Optional[pathlib.Path] = None,
) -> None:

    true_trig_nobjs_ = list(true_trig_nobjs.keys())
    true_trig_nobjs_counts = list(true_trig_nobjs.values())
    true_non_trig_nobjs_ = list(true_non_trig_nobjs.keys())
    true_non_trig_nobjs_counts = list(true_non_trig_nobjs.values())
    pred_trig_nobjs_ = list(pred_trig_nobjs.keys())
    pred_trig_nobjs_counts = list(pred_trig_nobjs.values())
    pred_non_trig_nobjs_ = list(pred_non_trig_nobjs.keys())
    pred_non_trig_nobjs_counts = list(pred_non_trig_nobjs.values())

    max_size = max(
        set(
            true_trig_nobjs_
            + true_non_trig_nobjs_
            + pred_trig_nobjs_
            + pred_non_trig_nobjs_
        )
    )

    fig, ax = plot_distributions(
        arrs=[
            true_trig_nobjs_,
            true_non_trig_nobjs_,
            pred_trig_nobjs_,
            pred_non_trig_nobjs_,
        ],
        bins=max_size,
        range=(0, max_size),
        weights=[
            true_trig_nobjs_counts,
            true_non_trig_nobjs_counts,
            pred_trig_nobjs_counts,
            pred_non_trig_nobjs_counts,
        ],
        hist_kwargs=[
            {
                "histtype": "stepfilled",
                "color": "blue",
                "alpha": 0.7,
                "label": r"$\mathrm{true\ trig}$",
            },
            {
                "histtype": "stepfilled",
                "color": "orange",
                "alpha": 0.7,
                "label": r"$\mathrm{true\ non-trig}$",
            },
            {
                "histtype": "stepfilled",
                "color": "green",
                "alpha": 0.7,
                "label": r"$\mathrm{pred\ trig}$",
            },
            {
                "histtype": "stepfilled",
                "color": "red",
                "alpha": 0.7,
                "label": r"$\mathrm{pred\ non-trig}$",
            },
        ],
        fig_kwargs={"figsize": (5, 4), "constrained_layout": True, "dpi": 300},
    )
    ax.set_xlabel(r"$\mathrm{Number\ of\ Objects}$")
    ax.set_ylabel(r"$\mathrm{Counts}$")
    ax.legend(
        loc="best",
        fontsize=10,
        frameon=False,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)


def get_metrics(
    results: Mapping[str, np.ndarray] | VtpHitOcInferenceResults, bkg_ids: list[int]
):
    if isinstance(results, VtpHitOcInferenceResults):
        results = results.to_dict()
    elif isinstance(results, Mapping):
        # required_fields = VtpHitOcInferenceResults.__annotations__.keys()
        # for field in required_fields:
        #     if field not in results:
        #         raise ValueError(f"Missing required field: {field}")
        pass

    trig_cm = np.zeros((2, 2), dtype=int)
    aggr_cm = np.zeros((2, 2), dtype=int)
    trig_aggr_cm = np.zeros((2, 2), dtype=int)
    clus_metrics = []
    trig_clus_metrics = []

    unique_events = np.unique(results["event_id"])

    for i in unique_events:
        mask = results["event_id"] == i
        result = {k: v[mask] for k, v in results.items()}

        trig_cm += confusion_matrix(
            result["cluster_type"],
            result["is_triggered"],
            labels=[0, 1],
        )

        is_bkg = np.isin(result["object_ids"], bkg_ids)
        num_unassigned = is_bkg.sum()

        object_ids = result["object_ids"]
        object_ids[is_bkg] = range(
            result["object_ids"].max() + 1,
            result["object_ids"].max() + 1 + num_unassigned,
        )

        truth_ids = result["truth_ids"]

        aggr_cm += pair_confusion_matrix(truth_ids, object_ids)
        clus_metrics.append(clustering_scores(truth_ids, object_ids))

        # aggregation confusion matrix gated on true triggered hits
        trig_aggr_cm += pair_confusion_matrix(
            truth_ids[result["cluster_type"] == 1],
            object_ids[result["cluster_type"] == 1],
        )
        trig_clus_metrics.append(
            clustering_scores(
                truth_ids[result["cluster_type"] == 1],
                object_ids[result["cluster_type"] == 1],
            )
        )

    hit_metrics = metrics_from_confusion(aggr_cm)

    clus_metrics = (
        {
            name: {
                "mean": float(np.mean([m[name] for m in clus_metrics])),
                "std": float(np.std([m[name] for m in clus_metrics])),
            }
            for name in clus_metrics[0].keys()
        }
        if clus_metrics
        else {}
    )

    trig_metrics = prediction_scores(
        truth=results["cluster_type"],
        prediction=results["is_triggered"],
        prob=results["trigger_probability"],
    )

    return {
        "num_events": len(unique_events),
        "num_hits": len(results["object_ids"]),
        "aggr_cm": aggr_cm,
        "trig_cm": trig_cm,
        "trig_aggr_cm": trig_aggr_cm,
        "clus_metrics": clus_metrics,
        "trig_clus_metrics": trig_clus_metrics,
        "hit_metrics": hit_metrics,
        "trig_metrics": trig_metrics,
    }


def collect_stats_summary(
    results: dict[str, np.ndarray] | VtpHitOcInferenceResults, bkg_ids: list[int]
):
    """
    Collect statistics summary for different object categories based on trigger status.

    Parameters
    ----------
    results : dict[str, np.ndarray]
        Dictionary containing the results with required fields.
    bkg_ids : list[int]
        List of background object IDs.

    Returns
    -------
    stats_summary : dict[str, Any]
        Dictionary containing statistics summary for different object categories, see `get_obj_stats`.

    """
    if isinstance(results, VtpHitOcInferenceResults):
        results = results.to_dict()

    required_fields = ["cluster_type", "is_triggered", "event_id", "object_ids"]
    for field in required_fields:
        if field not in results:
            raise ValueError(f"Missing required field: {field}")

    true_trig_mask = results["cluster_type"] == 1
    pred_trig_mask = results["is_triggered"] == 1
    true_non_trig_mask = results["cluster_type"] == 0
    pred_non_trig_mask = results["is_triggered"] == 0

    true_trig_obj_stats = get_obj_stats(
        results["event_id"][true_trig_mask],
        results["object_ids"][true_trig_mask],
        bkg_ids=bkg_ids,
    )

    pred_trig_obj_stats = get_obj_stats(
        results["event_id"][pred_trig_mask],
        results["object_ids"][pred_trig_mask],
        bkg_ids=bkg_ids,
    )

    true_all_obj_stats = get_obj_stats(
        results["event_id"],
        results["object_ids"],
        bkg_ids=bkg_ids,
    )
    pred_all_obj_stats = get_obj_stats(
        results["event_id"],
        results["object_ids"],
        bkg_ids=bkg_ids,
    )

    true_non_trig_obj_stats = get_obj_stats(
        results["event_id"][true_non_trig_mask],
        results["object_ids"][true_non_trig_mask],
        bkg_ids=bkg_ids,
    )

    pred_non_trig_obj_stats = get_obj_stats(
        results["event_id"][pred_non_trig_mask],
        results["object_ids"][pred_non_trig_mask],
        bkg_ids=bkg_ids,
    )

    stats_summary = {
        "true_trig": true_trig_obj_stats,
        "pred_trig": pred_trig_obj_stats,
        "true_all": true_all_obj_stats,
        "pred_all": pred_all_obj_stats,
        "true_non_trig": true_non_trig_obj_stats,
        "pred_non_trig": pred_non_trig_obj_stats,
    }
    return stats_summary


def generate_event_object_plots(
    event_ids: np.ndarray,
    truth_ids: np.ndarray,
    pred_ids: np.ndarray,
    pos: np.ndarray,
    bkg_ids: list[int],
    fig_dir: pathlib.Path,
    seed: int = 42,
    nplots: int = 10,
):
    """
    Helper function to generate nplots of truth vs pred hits on nps geometry.
    """
    from .report_helper import plot_event_objects

    unique_events = np.unique(event_ids)
    rng = np.random.default_rng(seed)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for i, evt_id in enumerate(
        rng.choice(unique_events, size=min(nplots, len(unique_events)), replace=False)
    ):
        mask = event_ids == evt_id
        plot_event_objects(
            truth_ids=truth_ids[mask],
            pred_ids=pred_ids[mask],
            pos=pos[mask],
            bkg_ids=bkg_ids,
            output_path=fig_dir / f"{i:04d}.png",
        )
