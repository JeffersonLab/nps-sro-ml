"""CSV, JSON, text, and optional plot output for baseline evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_csv_outputs(output_dir: Path, events: list[dict[str, Any]]) -> None:
    """Write one event row and one Hungarian-pair row to separate CSV files."""
    event_fields = [
        "event_id",
        "truth_clusters",
        "predicted_clusters",
        "matched_clusters",
        "missed_clusters",
        "fake_clusters",
        "mean_matched_iou",
        "mean_purity",
        "mean_efficiency",
        "split_truth_clusters",
        "merged_predicted_clusters",
    ]
    with (output_dir / "event_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows({key: event[key] for key in event_fields} for event in events)
    cluster_fields = [
        "event_id",
        "truth_cluster_id",
        "pred_cluster_id",
        "truth_size",
        "pred_size",
        "intersection",
        "iou",
        "dice",
        "purity",
        "efficiency",
        "matched",
    ]
    with (output_dir / "cluster_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cluster_fields)
        writer.writeheader()
        writer.writerows(pair for event in events for pair in event["pairs"])


def write_plots(
    output_dir: Path, metrics: dict[str, Any], events: list[dict[str, Any]]
) -> None:
    """Write diagnostics without coupling plotting to metric calculation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics["node_metrics"]["confusion_matrix"])
    fig, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=(0, 1),
        yticks=(0, 1),
        xticklabels=("Background", "Signal"),
        yticklabels=("Background", "Signal"),
        xlabel="Predicted",
        ylabel="Truth",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)

    successful = [p for event in events for p in event["pairs"] if p["matched"]]
    for field in ("iou", "purity", "efficiency"):
        fig, axis = plt.subplots()
        axis.hist([pair[field] for pair in successful], bins=20, range=(0, 1))
        axis.set(xlabel=field.capitalize(), ylabel="Matched cluster pairs")
        fig.tight_layout()
        fig.savefig(output_dir / f"{field}_distribution.png")
        plt.close(fig)
    fig, axis = plt.subplots()
    axis.scatter(
        [e["truth_clusters"] for e in events],
        [e["predicted_clusters"] for e in events],
        alpha=0.7,
    )
    maximum = max(
        [1] + [max(e["truth_clusters"], e["predicted_clusters"]) for e in events]
    )
    axis.plot([0, maximum], [0, maximum], "--", color="gray")
    axis.set(xlabel="Truth cluster count", ylabel="Predicted cluster count")
    fig.tight_layout()
    fig.savefig(output_dir / "true_vs_pred_cluster_count.png")
    plt.close(fig)


def write_results(
    output_dir: Path, metrics: dict[str, Any], events: list[dict[str, Any]]
) -> None:
    """Write the complete machine- and human-readable baseline report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
    write_csv_outputs(output_dir, events)
    node, cluster, compute = (
        metrics["node_metrics"],
        metrics["cluster_metrics"],
        metrics["compute"],
    )
    text = f"""=====================================================
NPS SRO ML — BASELINE OC ATTENTION EVALUATION
=====================================================
Checkpoint: {metrics['model']['checkpoint']}
Dataset: {metrics['dataset']['data_dir']} ({metrics['dataset']['events']} events)
Decoder: beta threshold {metrics['decoder']['beta_threshold']:.3f}; distance threshold {metrics['decoder']['distance_threshold']:.3f}

NODE SIGNAL/BACKGROUND PERFORMANCE
TP {node['tp']}  TN {node['tn']}  FP {node['fp']}  FN {node['fn']}
Accuracy {node['accuracy']:.6f}  Precision {node['precision']:.6f}  Recall {node['recall']:.6f}  F1 {node['f1']:.6f}
Specificity {node['specificity']:.6f}  FPR {node['fpr']:.6f}  FNR {node['fnr']:.6f}

CLUSTER RECONSTRUCTION PERFORMANCE
Truth {cluster['truth_clusters']}  Predicted {cluster['predicted_clusters']}  Matched {cluster['tp']}  Missed {cluster['fn']}  Fake {cluster['fp']}
Precision {cluster['precision']:.6f}  Recall {cluster['recall']:.6f}  F1 {cluster['f1']:.6f}
Mean/median IoU {cluster['mean_iou']:.6f}/{cluster['median_iou']:.6f}
Mean purity {cluster['mean_purity']:.6f}  efficiency {cluster['mean_efficiency']:.6f}  Dice {cluster['mean_dice']:.6f}
Split rate {cluster['split_rate']:.6f}  Merge rate {cluster['merge_rate']:.6f}

COMPUTATIONAL PERFORMANCE
Parameters {metrics['model']['total_parameters']} ({metrics['model']['trainable_parameters']} trainable)
Mean model latency {compute['mean_forward_latency_ms']:.3f} ms/event
Mean total latency {compute['mean_total_latency_ms']:.3f} ms/event; P95 {compute['p95_total_latency_ms']:.3f} ms/event
Throughput {compute['throughput_events_per_second']:.3f} events/s; Peak GPU memory {compute['peak_gpu_memory_mb']:.3f} MB
=====================================================
"""
    (output_dir / "summary.txt").write_text(text)
    write_plots(output_dir, metrics, events)
