from typing import Any

import numpy as np


def background_confusion_matrix(
    object_ids: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    true_background = object_ids < 0
    pred_background = cluster_ids < 0

    return np.array(
        [
            [
                np.sum(true_background & pred_background),
                np.sum(true_background & ~pred_background),
            ],
            [
                np.sum(~true_background & pred_background),
                np.sum(~true_background & ~pred_background),
            ],
        ],
        dtype=np.int64,
    )


def pairwise_cluster_confusion_matrix(
    event_ids: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    object_ids: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    pair_confusion = np.zeros((2, 2), dtype=np.int64)

    event_blocks = _group_labels_by_block(
        event_ids,
        det_x,
        det_y,
        object_ids,
        cluster_ids,
    )

    for block_map in event_blocks.values():
        block_labels = list(block_map.values())
        if len(block_labels) < 2:
            continue

        for left_idx in range(len(block_labels) - 1):
            left_true = block_labels[left_idx]["true"]
            left_pred = block_labels[left_idx]["pred"]
            for right_idx in range(left_idx + 1, len(block_labels)):
                right_true = block_labels[right_idx]["true"]
                right_pred = block_labels[right_idx]["pred"]

                # Position-aware and permutation-invariant: compare detector blocks and
                # treat them as matching when they share at least one object/cluster ID.
                true_same = bool(left_true.intersection(right_true))
                pred_same = bool(left_pred.intersection(right_pred))

                pair_confusion[int(true_same), int(pred_same)] += 1

    return pair_confusion


def _group_labels_by_block(
    event_ids: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    object_ids: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[int, dict[tuple[float, float], dict[str, set[int]]]]:
    events: dict[int, dict[tuple[float, float], dict[str, set[int]]]] = {}

    for event_id, x_pos, y_pos, object_id, cluster_id in zip(
        event_ids,
        det_x,
        det_y,
        object_ids,
        cluster_ids,
    ):
        event_key = int(event_id)
        block_key = (float(x_pos), float(y_pos))
        block_map = events.setdefault(event_key, {})
        label_sets = block_map.setdefault(block_key, {"true": set(), "pred": set()})

        if object_id >= 0:
            label_sets["true"].add(int(object_id))
        if cluster_id >= 0:
            label_sets["pred"].add(int(cluster_id))

    return events


def accuracy_from_confusion(confusion: np.ndarray) -> float:
    return _safe_div(np.trace(confusion), confusion.sum())


def precision_from_confusion(confusion: np.ndarray) -> float:
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    return _safe_div(tp, tp + fp)


def recall_from_confusion(confusion: np.ndarray) -> float:
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    return _safe_div(tp, tp + fn)


def f1_from_confusion(confusion: np.ndarray) -> float:
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    return _safe_f1(tp, fp, fn)


def summarize_clustering_metrics(
    event_ids: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    object_ids: np.ndarray,
    cluster_ids: np.ndarray,
    background_confusion: np.ndarray | None = None,
    pair_confusion: np.ndarray | None = None,
) -> dict[str, Any]:
    background_confusion = (
        background_confusion
        if background_confusion is not None
        else background_confusion_matrix(object_ids, cluster_ids)
    )
    pair_confusion = (
        pair_confusion
        if pair_confusion is not None
        else pairwise_cluster_confusion_matrix(
            event_ids,
            det_x,
            det_y,
            object_ids,
            cluster_ids,
        )
    )

    return {
        "num_events": int(np.unique(event_ids).size),
        "num_nodes": int(event_ids.size),
        "background_accuracy": accuracy_from_confusion(background_confusion),
        "background_precision": precision_from_confusion(background_confusion),
        "background_recall": recall_from_confusion(background_confusion),
        "background_f1": f1_from_confusion(background_confusion),
        "pairwise_accuracy": accuracy_from_confusion(pair_confusion),
        "pairwise_precision": precision_from_confusion(pair_confusion),
        "pairwise_recall": recall_from_confusion(pair_confusion),
        "pairwise_f1": f1_from_confusion(pair_confusion),
    }


def compute_clustering_metrics(
    event_ids: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    object_ids: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[str, Any]:
    background_confusion = background_confusion_matrix(object_ids, cluster_ids)
    pair_confusion = pairwise_cluster_confusion_matrix(
        event_ids,
        det_x,
        det_y,
        object_ids,
        cluster_ids,
    )
    summary = summarize_clustering_metrics(
        event_ids,
        det_x,
        det_y,
        object_ids,
        cluster_ids,
        background_confusion=background_confusion,
        pair_confusion=pair_confusion,
    )

    return {
        "summary": summary,
        "background_confusion": background_confusion,
        "pair_confusion": pair_confusion,
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _safe_f1(tp: float, fp: float, fn: float) -> float:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)
