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

    for event_id in np.unique(event_ids):
        event_mask = event_ids == event_id
        obj_ids = object_ids[event_mask]
        clus_ids = cluster_ids[event_mask]
        pos_x = det_x[event_mask]
        pos_y = det_y[event_mask]

        unique_pos, inverse = np.unique(
            np.stack([pos_x, pos_y], axis=1),
            axis=0,
            return_inverse=True,
        )

        block_obj_ids = np.full(unique_pos.shape[0], fill_value=-1, dtype=obj_ids.dtype)
        block_clus_ids = np.full(unique_pos.shape[0], fill_value=-1, dtype=clus_ids.dtype)

        for block_idx in range(unique_pos.shape[0]):
            block_mask = inverse == block_idx
            block_obj = np.unique(obj_ids[block_mask])
            block_clus = np.unique(clus_ids[block_mask])

            if block_obj.size > 1:
                raise ValueError(
                    "Inconsistent object IDs found for the same event/block position."
                )
            if block_clus.size > 1:
                raise ValueError(
                    "Inconsistent cluster IDs found for the same event/block position."
                )

            block_obj_ids[block_idx] = block_obj[0]
            block_clus_ids[block_idx] = block_clus[0]

        obj_ids = block_obj_ids
        clus_ids = block_clus_ids
        if obj_ids.size < 2:
            continue

        # Position-aware and permutation-invariant: compare detector blocks, not row order.
        true_same = (obj_ids[:, None] == obj_ids[None, :]) & (obj_ids[:, None] >= 0)
        pred_same = (clus_ids[:, None] == clus_ids[None, :]) & (clus_ids[:, None] >= 0)
        upper = np.triu_indices(obj_ids.size, k=1)
        true_same = true_same[upper]
        pred_same = pred_same[upper]

        pair_confusion[0, 0] += np.sum(~true_same & ~pred_same)
        pair_confusion[0, 1] += np.sum(~true_same & pred_same)
        pair_confusion[1, 0] += np.sum(true_same & ~pred_same)
        pair_confusion[1, 1] += np.sum(true_same & pred_same)

    return pair_confusion


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
