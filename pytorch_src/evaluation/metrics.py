"""Node classification and event-local cluster matching metrics.

Cluster labels are arbitrary: a predicted label of 7 can describe truth object 1.
Consequently clusters are paired by maximum IoU with the Hungarian algorithm,
never by comparing their integer labels directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Return a finite ratio, using zero when the ratio is undefined."""
    return float(numerator / denominator) if denominator else 0.0


def node_binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Compute signal/background metrics (signal means cluster ID is nonzero).

    TP is a signal node reconstructed as signal; TN is background reconstructed
    as background. FP and FN are respectively background promoted to signal and
    signal rejected as background. Precision=TP/(TP+FP), recall=TP/(TP+FN),
    specificity=TN/(TN+FP), FPR=FP/(FP+TN), and FNR=FN/(FN+TP).
    """
    truth = np.asarray(truth).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError(
            f"Node arrays are not aligned: {truth.shape} != {prediction.shape}"
        )
    truth_signal, pred_signal = truth != 0, prediction != 0
    tp = int(np.sum(truth_signal & pred_signal))
    tn = int(np.sum(~truth_signal & ~pred_signal))
    fp = int(np.sum(~truth_signal & pred_signal))
    fn = int(np.sum(truth_signal & ~pred_signal))
    precision, recall = safe_divide(tp, tp + fp), safe_divide(tp, tp + fn)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": safe_divide(tp + tn, truth.size),
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "specificity": safe_divide(tn, tn + fp),
        "fpr": safe_divide(fp, fp + tn),
        "fnr": safe_divide(fn, fn + tp),
    }


@dataclass(frozen=True)
class ClusterPair:
    """Quality values for one Hungarian-assigned truth/prediction pair."""

    event_id: int
    truth_cluster_id: int
    pred_cluster_id: int
    truth_size: int
    pred_size: int
    intersection: int
    iou: float
    dice: float
    purity: float
    efficiency: float
    matched: bool


def match_event_clusters(
    truth: np.ndarray,
    prediction: np.ndarray,
    event_id: int,
    iou_threshold: float = 0.5,
    overlap_threshold: float = 0.1,
) -> dict[str, Any]:
    """Match non-background clusters within one event and diagnose splits/merges.

    IoU is intersection/union; purity is intersection/predicted size; efficiency
    is intersection/truth size; Dice is twice the intersection divided by the
    sum of sizes. Hungarian assignment maximizes total IoU one-to-one. An
    assigned pair is a cluster TP only when its IoU reaches ``iou_threshold``;
    every other predicted/truth cluster contributes cluster FP/FN respectively.

    For exploratory split/merge diagnostics, a pair significantly overlaps when
    ``intersection / min(truth_size, predicted_size) >= overlap_threshold``.
    Thus a truth cluster overlapping multiple predictions is split, and a
    prediction overlapping multiple truths is merged.
    """
    if not 0 <= iou_threshold <= 1 or not 0 <= overlap_threshold <= 1:
        raise ValueError("IoU and split/merge thresholds must be in [0, 1]")
    truth, prediction = np.asarray(truth).reshape(-1), np.asarray(prediction).reshape(
        -1
    )
    if truth.shape != prediction.shape:
        raise ValueError("Truth and prediction must describe identical nodes")
    truth_ids = np.unique(truth[truth != 0]).astype(int)
    pred_ids = np.unique(prediction[prediction != 0]).astype(int)
    intersections = np.zeros((len(pred_ids), len(truth_ids)), dtype=int)
    ious = np.zeros_like(intersections, dtype=float)
    overlap = np.zeros_like(intersections, dtype=float)
    pred_sizes = np.array([np.sum(prediction == p) for p in pred_ids])
    truth_sizes = np.array([np.sum(truth == t) for t in truth_ids])
    for pi, pred_id in enumerate(pred_ids):
        for ti, truth_id in enumerate(truth_ids):
            intersection = int(np.sum((prediction == pred_id) & (truth == truth_id)))
            intersections[pi, ti] = intersection
            union = pred_sizes[pi] + truth_sizes[ti] - intersection
            ious[pi, ti] = safe_divide(intersection, union)
            overlap[pi, ti] = safe_divide(
                intersection, min(pred_sizes[pi], truth_sizes[ti])
            )

    pairs: list[ClusterPair] = []
    if ious.size:
        pred_indices, truth_indices = linear_sum_assignment(ious, maximize=True)
        for pi, ti in zip(pred_indices, truth_indices):
            intersection = int(intersections[pi, ti])
            pairs.append(
                ClusterPair(
                    event_id,
                    int(truth_ids[ti]),
                    int(pred_ids[pi]),
                    int(truth_sizes[ti]),
                    int(pred_sizes[pi]),
                    intersection,
                    float(ious[pi, ti]),
                    safe_divide(2 * intersection, pred_sizes[pi] + truth_sizes[ti]),
                    safe_divide(intersection, pred_sizes[pi]),
                    safe_divide(intersection, truth_sizes[ti]),
                    bool(ious[pi, ti] >= iou_threshold),
                )
            )
    successful = [pair for pair in pairs if pair.matched]
    tp = len(successful)
    split_count = int(np.sum(np.sum(overlap >= overlap_threshold, axis=0) > 1))
    merge_count = int(np.sum(np.sum(overlap >= overlap_threshold, axis=1) > 1))
    return {
        "event_id": event_id,
        "truth_clusters": len(truth_ids),
        "predicted_clusters": len(pred_ids),
        "matched_clusters": tp,
        "missed_clusters": len(truth_ids) - tp,
        "fake_clusters": len(pred_ids) - tp,
        "split_truth_clusters": split_count,
        "merged_predicted_clusters": merge_count,
        "pairs": [asdict(pair) for pair in pairs],
        "mean_matched_iou": _mean([p.iou for p in successful]),
        "mean_purity": _mean([p.purity for p in successful]),
        "mean_efficiency": _mean([p.efficiency for p in successful]),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
    }


def aggregate_cluster_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate event-local cluster TP/FP/FN and successful-pair qualities."""
    tp = sum(event["matched_clusters"] for event in events)
    fp = sum(event["fake_clusters"] for event in events)
    fn = sum(event["missed_clusters"] for event in events)
    truth_total = sum(event["truth_clusters"] for event in events)
    pred_total = sum(event["predicted_clusters"] for event in events)
    pairs = [pair for event in events for pair in event["pairs"] if pair["matched"]]
    precision, recall = safe_divide(tp, tp + fp), safe_divide(tp, tp + fn)
    distributions = {
        name: _distribution([p[name] for p in pairs])
        for name in ("iou", "dice", "purity", "efficiency")
    }
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "truth_clusters": truth_total,
        "predicted_clusters": pred_total,
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        **{
            f"{stat}_{name}": values[stat]
            for name, values in distributions.items()
            for stat in ("mean", "median", "std")
        },
        "split_truth_clusters": sum(e["split_truth_clusters"] for e in events),
        "merged_predicted_clusters": sum(
            e["merged_predicted_clusters"] for e in events
        ),
        "split_rate": safe_divide(
            sum(e["split_truth_clusters"] for e in events), truth_total
        ),
        "merge_rate": safe_divide(
            sum(e["merged_predicted_clusters"] for e in events), pred_total
        ),
    }
