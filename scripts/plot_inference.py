#!/usr/bin/env python3

import argparse
import json
import pathlib
import re
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
        cluster_sizes = df_clus.groupby(["event_id", "cluster_ids"]).size().to_numpy()
        _plot_cluster_size_distribution(
            cluster_sizes,
            plot_dir / "cluster_size_distribution.png",
        )

    metrics = compute_clustering_metrics(
        event_ids=results["event_id"].to_numpy(),
        det_x=results["det_x"].to_numpy(),
        det_y=results["det_y"].to_numpy(),
        object_ids=results["object_ids"].to_numpy(),
        cluster_ids=results["cluster_ids"].to_numpy(),
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

    pulse_results = _extract_pulse_level_results(results)
    if pulse_results is None or pulse_results.empty:
        return

    finite_pulse_min_d = pulse_results.loc[
        np.isfinite(pulse_results["pulse_min_d"]),
        "pulse_min_d",
    ].to_numpy()
    if finite_pulse_min_d.size > 0:
        _plot_min_distance_distribution(
            finite_pulse_min_d,
            plot_dir / "pulse_min_distance_distribution.png",
        )

    _plot_beta_distribution(
        pulse_results["pulse_beta"].to_numpy(),
        plot_dir / "pulse_beta_distribution.png",
    )
    _plot_score_distribution(
        pulse_results["pulse_score"].to_numpy(),
        plot_dir / "pulse_score_distribution.png",
    )

    df_pulse_clus = pulse_results[pulse_results["pulse_cluster_ids"] >= 0]
    if not df_pulse_clus.empty:
        pulse_cluster_sizes = (
            df_pulse_clus.groupby(["event_id", "pulse_cluster_ids"]).size().to_numpy()
        )
        _plot_cluster_size_distribution(
            pulse_cluster_sizes,
            plot_dir / "pulse_cluster_size_distribution.png",
        )

    pulse_metrics = compute_clustering_metrics(
        event_ids=pulse_results["event_id"].to_numpy(),
        det_x=pulse_results["det_x"].to_numpy(),
        det_y=pulse_results["det_y"].to_numpy(),
        object_ids=pulse_results["pulse_object_ids"].to_numpy(),
        cluster_ids=pulse_results["pulse_cluster_ids"].to_numpy(),
    )
    _write_metrics_json(pulse_metrics, metric_dir / "pulse_metrics.json")

    _plot_confusion_matrix(
        matrix=pulse_metrics["background_confusion"],
        row_labels=["true background", "true signal"],
        col_labels=["pred background", "pred signal"],
        title="Pulse Background Recognition",
        pth=plot_dir / "pulse_background_confusion_matrix.png",
    )
    _plot_confusion_matrix(
        matrix=pulse_metrics["pair_confusion"],
        row_labels=["true different", "true same"],
        col_labels=["pred different", "pred same"],
        title="Pulse Cluster Recognition (Strict)",
        pth=plot_dir / "pulse_pair_confusion_matrix.png",
    )
    _plot_confusion_matrix(
        matrix=pulse_metrics["pair_confusion_overlap_tolerant"],
        row_labels=["true different", "true same"],
        col_labels=["pred different", "pred same"],
        title="Pulse Cluster Recognition (Overlap Tolerant)",
        pth=plot_dir / "pulse_pair_confusion_overlap_tolerant_matrix.png",
    )


def _extract_pulse_level_results(results: pd.DataFrame) -> pd.DataFrame | None:
    pulse_slots = _find_slot_indices(results.columns, "pulse_cluster_ids")
    if not pulse_slots:
        return None

    pulse_records: list[pd.DataFrame] = []
    for slot in pulse_slots:
        slot_frame = pd.DataFrame(
            {
                "event_id": results["event_id"].to_numpy(),
                "det_x": results["det_x"].to_numpy(),
                "det_y": results["det_y"].to_numpy(),
                "slot": np.full(len(results), slot, dtype=np.int64),
                "pulse_cluster_ids": _slot_column_or_default(
                    results, f"pulse_cluster_ids_{slot}", -1
                ),
                "pulse_min_d": _slot_column_or_default(
                    results, f"pulse_min_d_{slot}", np.inf
                ),
                "pulse_beta": _slot_column_or_default(results, f"pulse_beta_{slot}", 0.0),
                "pulse_score": _slot_column_or_default(results, f"pulse_score_{slot}", 0.0),
                "pulse_object_ids": _slot_column_or_default(
                    results, f"pulse_object_ids_{slot}", -1
                ),
            }
        )
        pulse_records.append(slot_frame)

    pulse_results = pd.concat(pulse_records, ignore_index=True)

    active_mask = (
        (pulse_results["pulse_object_ids"] >= 0)
        | (pulse_results["pulse_cluster_ids"] >= 0)
        | (pulse_results["pulse_score"] > 0.0)
        | (pulse_results["pulse_beta"] > 0.0)
        | np.isfinite(pulse_results["pulse_min_d"])
    )
    return pulse_results.loc[active_mask].reset_index(drop=True)


def _find_slot_indices(columns: Iterable[str], prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    slots: list[int] = []
    for column in columns:
        match = pattern.match(column)
        if match is not None:
            slots.append(int(match.group(1)))
    return sorted(set(slots))


def _slot_column_or_default(
    results: pd.DataFrame,
    column: str,
    default: float | int,
) -> np.ndarray:
    if column in results.columns:
        return results[column].to_numpy()
    return np.full(len(results), default)


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


def _plot_score_distribution(score_arr: np.ndarray, pth: pathlib.Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True, dpi=300)
    ax.hist(
        score_arr,
        bins=100,
        range=(0, 1),
        histtype="stepfilled",
        color="blue",
        alpha=0.7,
    )
    ax.set_xlabel(r"$\mathrm{pulse} \ \mathrm{score}$", fontsize=14)
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
