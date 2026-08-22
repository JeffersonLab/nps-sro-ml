"""Tests for model-independent baseline cluster evaluation."""

# ruff: noqa: D103

import numpy as np
import pytest

from evaluation.metrics import (
    aggregate_cluster_metrics,
    match_event_clusters,
    node_binary_metrics,
)


def evaluate_event(truth, prediction, iou_threshold=0.5, overlap_threshold=0.1):
    event = match_event_clusters(
        np.asarray(truth), np.asarray(prediction), 0, iou_threshold, overlap_threshold
    )
    return event, aggregate_cluster_metrics([event])


def test_perfect_reconstruction():
    truth = np.array([0, 1, 1, 2, 2])
    event, metrics = evaluate_event(truth, truth)
    node = node_binary_metrics(truth, truth)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 1
    assert metrics["mean_iou"] == metrics["mean_purity"] == 1
    assert metrics["mean_efficiency"] == metrics["mean_dice"] == 1
    assert node["accuracy"] == node["f1"] == 1
    assert event["matched_clusters"] == 2


def test_completely_missed_cluster():
    _, metrics = evaluate_event([1, 1, 1], [0, 0, 0])
    assert metrics["fn"] == 1
    assert metrics["recall"] == 0


def test_fake_cluster():
    _, metrics = evaluate_event([0, 0, 0], [4, 4, 4])
    assert metrics["fp"] == 1


def test_split_diagnostic():
    event, _ = evaluate_event([1, 1, 1, 1], [7, 7, 4, 4])
    assert event["split_truth_clusters"] == 1


def test_merge_diagnostic():
    event, _ = evaluate_event([1, 1, 2, 2], [7, 7, 7, 7])
    assert event["merged_predicted_clusters"] == 1


def test_arbitrary_cluster_labels_are_perfect():
    _, metrics = evaluate_event([1, 1, 1, 2, 2, 2], [7, 7, 7, 4, 4, 4])
    assert metrics["tp"] == 2
    assert metrics["mean_iou"] == 1


def test_events_are_matched_independently():
    # The same predicted label is deliberately reused in both events. Event-local
    # matching correctly counts two reconstructed objects, not one global object.
    first = match_event_clusters(np.array([1, 1]), np.array([7, 7]), 0)
    second = match_event_clusters(np.array([2, 2]), np.array([7, 7]), 1)
    metrics = aggregate_cluster_metrics([first, second])
    assert metrics["tp"] == 2
    assert {
        pair["event_id"] for event in (first, second) for pair in event["pairs"]
    } == {0, 1}


def test_alignment_mismatch_is_rejected():
    with pytest.raises(ValueError, match="identical nodes"):
        match_event_clusters(np.array([1]), np.array([1, 1]), 0)
