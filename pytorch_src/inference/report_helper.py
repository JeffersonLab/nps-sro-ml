import pathlib
from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_MPL_CONFIG = {
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

mpl.rcParams.update(_MPL_CONFIG)


def plot_confusion_matrix(
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: Optional[str] = None,
    output_path: Optional[pathlib.Path] = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot the confusion matrix as a heatmap with row percentages.

    Parameters
    ----------
    matrix : np.ndarray
        Confusion matrix of shape (num_classes, num_classes).
    row_labels : list[str]
        Labels for the rows of the confusion matrix.
    column_labels : list[str]
        Labels for the columns of the confusion matrix.
    title : Optional[str], optional
        Title of the plot, by default None.
    output_path : Optional[pathlib.Path], optional
        Path to save the output plot, by default None.

    Returns
    -------
    tuple[mpl.figure.Figure, mpl.axes.Axes]
        The figure and axes objects of the plot.

    """

    matrix = np.asarray(matrix, dtype=np.int64)
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix * 100.0,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True, dpi=220)
    image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(image, ax=ax, label="Row percentage [%]")
    ax.set_xticks(range(len(column_labels)), labels=column_labels)
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.tick_params(axis="x", rotation=18)
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if percentages[row, column] >= 55 else "black"
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{percentages[row, column]:.1f}%",
                ha="center",
                va="center",
                color=color,
            )
    if output_path is not None:
        fig.savefig(output_path)
        plt.close(fig)
    return fig, ax


def plot_event_objects(
    truth_ids: np.ndarray,
    pred_ids: np.ndarray,
    pos: np.ndarray,
    empty_idx: int,
    output_path: pathlib.Path,
) -> None:
    """
    Plot the truth and predicted event objects on a 2D grid.

    Parameters
    ----------
    truth_ids : np.ndarray
        Array of truth object IDs.
    pred_ids : np.ndarray
        Array of predicted object IDs.
    pos : np.ndarray
        Array of object positions with shape (N, 2).
    empty_idx : int
        Index representing empty positions.
    output_path : pathlib.Path
        Path to save the output plot.

    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)

    for ax in axes:
        _draw_nps_frame(ax)

    _draw_detector_hits(
        axes[0],
        truth_ids,
        pos,
        ignored_id=None,
        noise_idx=[empty_idx],
    )
    _draw_detector_hits(
        axes[1],
        pred_ids,
        pos,
        ignored_id=None,
        noise_idx=[empty_idx],
    )
    axes[0].set_title(r"$\mathrm{Truth Objects}$", fontsize=12)
    axes[1].set_title(r"$\mathrm{Predicted Objects}$", fontsize=12)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _draw_detector_hits(
    ax: plt.Axes,
    object_ids: np.ndarray,
    positions: np.ndarray,
    noise_idx: list[int] = [],
    ignored_id: Optional[int] = None,
    cmap: Optional[mpl.colors.Colormap] = None,
) -> None:

    if cmap is None:
        cmap = mpl.colormaps["rainbow"]

    if ignored_id is not None:
        mask = object_ids != ignored_id
        object_ids = object_ids[mask]
        positions = positions[mask]

    obj_mask = ~np.isin(object_ids, noise_idx)
    unique_ids = np.unique(object_ids[obj_mask])

    colors = cmap(np.linspace(0, 1, len(unique_ids)))

    for i, oid in enumerate(unique_ids):
        mask = object_ids == oid
        x = positions[mask, 0]
        y = positions[mask, 1]
        patch = mpl.patches.Rectangle(
            xy=(x[0] - 0.5, y[0] - 0.5),
            width=1.0,
            height=1.0,
            facecolor=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.add_patch(patch)

    for i in noise_idx:
        mask = object_ids == i
        x = positions[mask, 0]
        y = positions[mask, 1]
        if len(x) == 0 or len(y) == 0:
            continue
        patch = mpl.patches.Rectangle(
            xy=(x[0] - 0.5, y[0] - 0.5),
            width=1.0,
            height=1.0,
            facecolor="gray",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.add_patch(patch)


def _draw_nps_frame(
    ax: plt.Axes,
) -> None:

    from datasets.nps import NCOLS, NROWS

    for i in range(NCOLS):
        ax.axvline(x=i - 0.5, color="gray", linewidth=0.35, alpha=0.8)
    for j in range(NROWS):
        ax.axhline(y=j - 0.5, color="gray", linewidth=0.35, alpha=0.8)

    ax.set_xlim(-0.5, NCOLS - 0.5)
    ax.set_ylim(-0.5, NROWS - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Column}$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Row}$", fontsize=12)
    ax.grid(color="0.85", linewidth=0.35)
