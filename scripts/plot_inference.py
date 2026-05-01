#!/usr/bin/env python3

import argparse
import pathlib
import sys
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pytorch_src"))

from inference.metrics import compute_clustering_metrics

NCOLS = 30
NROWS = 36

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

    df_clus = results[results["cluster_ids"] >= 0]
    if not df_clus.empty:
        sampled_events = _sample_events(results["event_id"].unique(), num_events, seed)
        df_clus_sampled = df_clus[df_clus["event_id"].isin(sampled_events)]
        if not df_clus_sampled.empty:
            cluster_sizes = (
                df_clus_sampled.groupby(["event_id", "cluster_ids"]).size().to_numpy()
            )
            _plot_cluster_size_distribution(
                cluster_sizes,
                plot_dir / "cluster_size_distribution.png",
                sampled_events=sampled_events,
            )

    metrics = compute_clustering_metrics(
        event_ids=results["event_id"].to_numpy(),
        det_x=results["det_x"].to_numpy(),
        det_y=results["det_y"].to_numpy(),
        object_ids=results["object_ids"].to_numpy(),
        cluster_ids=results["cluster_ids"].to_numpy(),
    )
    pd.DataFrame([metrics["summary"]]).to_csv(metric_dir / "summary.csv", index=False)
    pd.DataFrame(
        metrics["background_confusion"],
        index=["true_background", "true_signal"],
        columns=["pred_background", "pred_signal"],
    ).to_csv(metric_dir / "background_confusion_matrix.csv")
    pd.DataFrame(
        metrics["pair_confusion"],
        index=["true_different", "true_same"],
        columns=["pred_different", "pred_same"],
    ).to_csv(metric_dir / "pair_confusion_matrix.csv")

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
        title="Permutation-Invariant Cluster Recognition",
        pth=plot_dir / "pair_confusion_matrix.png",
    )
def _sample_events(
    event_ids: Iterable[int] | np.ndarray,
    num_events: int,
    seed: int,
) -> np.ndarray:
    event_ids = np.sort(np.asarray(event_ids))
    num_events = min(num_events, len(event_ids))
    if num_events == 0:
        return np.array([], dtype=event_ids.dtype)
    if len(event_ids) <= num_events:
        return event_ids

    rng = np.random.default_rng(seed)
    sampled = rng.choice(event_ids, size=num_events, replace=False)
    return np.sort(sampled)
def _plot_confusion_matrix(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    pth: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True, dpi=300)
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(col_labels)), labels=col_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_title(title, fontsize=13)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")

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
    sampled_events: np.ndarray,
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
    ax.set_title(
        f"Sampled events: {', '.join(str(int(event_id)) for event_id in sampled_events)}",
        fontsize=10,
    )
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
