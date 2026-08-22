#!/usr/bin/env python3

import argparse
import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pytorch_src"))

from inference.metrics import compute_clustering_metrics

PLOT_CONFIG = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "figure.dpi": 120,
    "figure.figsize": (4, 3.5),
    "figure.facecolor": "white",
    "xtick.top": True,
    "xtick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.right": True,
    "ytick.direction": "in",
    "ytick.minor.visible": True,
}
for key, value in PLOT_CONFIG.items():
    mpl.rcParams[key] = value


def main(
    input_path: pathlib.Path,
    out_dir: pathlib.Path | None = None,
    num_events: int = 5,
    seed: int = 0,
) -> None:
    results = pd.read_csv(input_path)
    resolved_out_dir = out_dir or input_path.parent
    report(results, resolved_out_dir, num_events, seed)


def report(
    results: pd.DataFrame,
    out_dir: pathlib.Path,
    num_events: int,
    seed: int,
) -> None:
    if results.empty:
        return

    plot_dir = out_dir / "plots"
    metric_dir = out_dir / "metrics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    _plot_beta_distribution(results["beta"].to_numpy(), plot_dir / "beta_distribution.png")

    finite_min_d = results.loc[np.isfinite(results["min_d"]), "min_d"].to_numpy()
    if finite_min_d.size > 0:
        _plot_min_distance_distribution(
            finite_min_d,
            plot_dir / "min_distance_distribution.png",
        )

    df_clus = results[results["object_ids"] >= 0]
    if not df_clus.empty:
        cluster_sizes = df_clus.groupby(["event_id", "object_ids"]).size().to_numpy()
        _plot_cluster_size_distribution(
            cluster_sizes,
            plot_dir / "cluster_size_distribution.png",
        )

    metrics = compute_clustering_metrics(
        event_ids=results["event_id"].to_numpy(),
        det_x=results["pos_0"].to_numpy(),
        det_y=results["pos_1"].to_numpy(),
        object_ids=results["truth_ids"].to_numpy(),
        cluster_ids=results["object_ids"].to_numpy(),
    )
    _write_metrics_json(metrics, metric_dir / "metrics.json")

    _plot_confusion_matrix(
        matrix=metrics["background_confusion"],
        row_labels=["true background", "true signal"],
        col_labels=["pred background", "pred signal"],
        title="Background Recognition",
        pth=plot_dir / "background_confusion_matrix.png",
    )
    _plot_confusion_matrix(
        matrix=metrics["pair_confusion"],
        row_labels=["true different", "true same"],
        col_labels=["pred different", "pred same"],
        title="Permutation-Invariant Cluster Recognition (Strict)",
        pth=plot_dir / "pair_confusion_matrix.png",
    )
    _plot_confusion_matrix(
        matrix=metrics["pair_confusion_overlap_tolerant"],
        row_labels=["true different", "true same"],
        col_labels=["pred different", "pred same"],
        title="Permutation-Invariant Cluster Recognition (Overlap Tolerant)",
        pth=plot_dir / "pair_confusion_overlap_tolerant_matrix.png",
    )


def _write_metrics_json(metrics: dict, pth: pathlib.Path) -> None:
    background_percent = _confusion_percentages(metrics["background_confusion"])
    pair_percent = _confusion_percentages(metrics["pair_confusion"])
    pair_overlap_tolerant_percent = _confusion_percentages(
        metrics["pair_confusion_overlap_tolerant"]
    )
    serializable = {
        "summary": metrics["summary"],
        "background_confusion": {
            "counts": metrics["background_confusion"].tolist(),
            "percentages": background_percent.tolist(),
        },
        "pair_confusion": {
            "counts": metrics["pair_confusion"].tolist(),
            "percentages": pair_percent.tolist(),
        },
        "pair_confusion_overlap_tolerant": {
            "counts": metrics["pair_confusion_overlap_tolerant"].tolist(),
            "percentages": pair_overlap_tolerant_percent.tolist(),
        },
    }
    pth.write_text(json.dumps(serializable, indent=2) + "\n")


def _confusion_percentages(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    if total == 0:
        return np.zeros_like(matrix, dtype=float)
    return matrix.astype(float) * 100.0 / total


def _plot_confusion_matrix(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    pth: pathlib.Path,
) -> None:
    percentages = _confusion_percentages(matrix)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True, dpi=300)
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(col_labels)), labels=col_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_title(title, fontsize=13)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j])}\n{percentages[i, j]:.1f}%",
                ha="center",
                va="center",
                color="black",
            )

    fig.savefig(pth)
    plt.close(fig)


def _plot_beta_distribution(beta_arr: np.ndarray, pth: pathlib.Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True, dpi=300)
    ax.hist(
        beta_arr,
        bins=100,
        range=(0, 1),
        histtype="stepfilled",
        color="blue",
        alpha=0.7,
    )
    ax.set_xlabel(r"$\mathrm{seedness} \ \beta$", fontsize=14)
    ax.set_ylabel(r"$\mathrm{Counts}$", fontsize=14)
    fig.savefig(pth)
    plt.close(fig)


def _plot_min_distance_distribution(min_d_arr: np.ndarray, pth: pathlib.Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True, dpi=300)
    ax.hist(
        min_d_arr,
        bins=100,
        range=(0, 10),
        histtype="stepfilled",
        color="blue",
        alpha=0.7,
    )
    ax.set_xlabel(r"$\mathrm{min} \ d_{\mathrm{seed}}$", fontsize=14)
    ax.set_ylabel(r"$\mathrm{Counts}$", fontsize=14)
    fig.savefig(pth)
    plt.close(fig)


def _plot_cluster_size_distribution(
    cluster_size_arr: np.ndarray,
    pth: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True, dpi=300)
    ax.hist(
        cluster_size_arr,
        bins=30,
        range=(0, 30),
        histtype="stepfilled",
        color="blue",
        alpha=0.7,
    )
    ax.set_xlabel(r"$\mathrm{Cluster} \ \mathrm{Size}$", fontsize=14)
    ax.set_ylabel(r"$\mathrm{Counts}$", fontsize=14)
    fig.savefig(pth)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot OC inference results and compute clustering metrics."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path to inference results CSV.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Directory for plots and metrics. Defaults to the CSV parent directory.",
    )
    parser.add_argument(
        "-n",
        "--num-events",
        type=int,
        default=5,
        help="Number of random events to use for the cluster-size distribution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for event sampling.",
    )
    cli_args = parser.parse_args()
    main(
        input_path=cli_args.input,
        out_dir=cli_args.out_dir,
        num_events=cli_args.num_events,
        seed=cli_args.seed,
    )
