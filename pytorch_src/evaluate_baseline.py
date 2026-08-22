#!/usr/bin/env python3
"""Evaluate an existing Object Condensation attention checkpoint on held-out data."""

from __future__ import annotations

import argparse
import pathlib
import random
import time
from typing import Any

import numpy as np
import torch

from datasets.nps import NPSDataLoader
from evaluation.metrics import (
    aggregate_cluster_metrics,
    match_event_clusters,
    node_binary_metrics,
)
from evaluation.reporting import write_results
from models.oc_attn import ObjectCondensationAttn
from models.oc_inference import oc_inference_per_batch
from utils.graph import find_connected_components_undirected

MODEL_ARGUMENTS = {
    "in_feats",
    "pos_dim",
    "wf_embed_dim",
    "wf_lstm_hidden",
    "wf_lstm_layers",
    "wf_lstm_dropout",
    "wf_out_dim",
    "n_gravnet_layers",
    "gravnet_knn",
    "d_model",
    "n_enc_layers",
    "num_heads",
    "attn_dropout",
    "attn_ff",
    "oc_mlp_pos_hidden",
    "oc_mlp_dropout",
    "oc_mlp_beta_hidden",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line paths, thresholds, reproducibility, and timing options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--data-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/baseline_oc_attention"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta-thres", type=float, default=0.4)
    parser.add_argument("--dist-thres", type=float, default=0.8)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--split-merge-overlap-threshold", type=float, default=0.1)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    return parser.parse_args()


def load_model(
    checkpoint_path: pathlib.Path, device: torch.device
) -> tuple[ObjectCondensationAttn, dict[str, Any]]:
    """Reconstruct the exact saved architecture and strictly load its weights."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not load checkpoint {checkpoint_path}: {error}"
        ) from error
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected a BaseTrainer checkpoint containing 'state_dict'")
    if checkpoint.get("arch") != "ObjectCondensationAttn":
        raise ValueError(
            f"Unsupported checkpoint architecture: {checkpoint.get('arch')!r}"
        )
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            "Checkpoint has no model metadata; architecture arguments are unavailable"
        )
    kwargs = {key: metadata[key] for key in MODEL_ARGUMENTS if key in metadata}
    missing = MODEL_ARGUMENTS - kwargs.keys()
    if missing:
        raise ValueError(
            f"Checkpoint model metadata is incomplete; missing: {sorted(missing)}"
        )
    config = checkpoint.get("config", {})
    if isinstance(config, dict) and config.get("scaler"):
        raise ValueError(
            "Checkpoint indicates input scaling, but BaseTrainer did not serialize scaler state"
        )
    model = ObjectCondensationAttn(**kwargs).to(device)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint weights do not match ObjectCondensationAttn: {error}"
        ) from error
    model.eval()
    return model, checkpoint


def truth_from_graph(graph: Any) -> torch.Tensor:
    """Reproduce ObjectCondensationTrainer's connected-component object IDs."""
    truth = torch.zeros(graph.num_nodes, dtype=torch.long, device=graph.x.device)
    for cluster_id, nodes in enumerate(
        find_connected_components_undirected(graph.num_nodes, graph.edge_index), start=1
    ):
        truth[nodes] = cluster_id
    return truth


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run aligned inference, existing OC decoding, matching, and aggregation."""
    if not args.data_dir.is_dir():
        raise NotADirectoryError(f"Data directory does not exist: {args.data_dir}")
    paths = sorted(args.data_dir.glob("*.pt"))
    if not paths:
        raise ValueError(f"No .pt graph files found in {args.data_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, _ = load_model(args.checkpoint, device)
    loader = NPSDataLoader(
        data_paths=paths,
        shuffle=False,
        batch_size=args.batch_size,
        validation_split=0.0,
        num_workers=args.num_workers,
    )
    if loader.num_features_ != model.in_feats:
        raise ValueError(
            f"Input feature mismatch: data has {loader.num_features_}, checkpoint expects {model.in_feats}"
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    all_truth: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    forward_per_event: list[float] = []
    total_per_event: list[float] = []
    event_id = 0
    with torch.no_grad():
        for batch_index, data in enumerate(loader):
            data = data.to(device)
            mask = torch.ones(data.x.size(0), dtype=torch.bool, device=device)
            for _ in range(args.warmup_iterations if batch_index == 0 else 0):
                model(data.x, data.pos, batch=data.batch, mask=mask)
            _synchronize(device)
            start = time.perf_counter()
            # x_c is the learned clustering coordinate. beta is an OC seed strength,
            # not a calibrated signal probability. The repository decoder selects
            # beta > beta_thres seeds, assigns same-event nodes to their nearest seed,
            # and rejects assignments farther than dist_thres.
            x_c, beta = model(data.x, data.pos, batch=data.batch, mask=mask)
            _synchronize(device)
            forward_end = time.perf_counter()
            prediction, _ = oc_inference_per_batch(
                x_c,
                beta,
                data.batch[mask],
                beta_thres=args.beta_thres,
                dist_thres=args.dist_thres,
                bkg_idx=0,
            )
            _synchronize(device)
            total_end = time.perf_counter()
            graph_count = int(data.num_graphs)
            forward_per_event.extend(
                [(forward_end - start) * 1000 / graph_count] * graph_count
            )
            total_per_event.extend(
                [(total_end - start) * 1000 / graph_count] * graph_count
            )

            truth = torch.cat(
                [truth_from_graph(graph) for graph in data.to_data_list()]
            )
            aligned_batch = data.batch[mask]
            assert (
                prediction.numel() == truth.numel()
            ), "Prediction/truth node alignment failed"
            assert (
                aligned_batch.numel() == truth.numel()
            ), "Batch/truth node alignment failed"
            truth_np, pred_np, batch_np = (
                tensor.detach().cpu().numpy()
                for tensor in (truth, prediction, aligned_batch)
            )
            all_truth.append(truth_np)
            all_prediction.append(pred_np)
            for local_id in np.unique(batch_np):
                event_mask = batch_np == local_id
                events.append(
                    match_event_clusters(
                        truth_np[event_mask],
                        pred_np[event_mask],
                        event_id,
                        args.match_iou_threshold,
                        args.split_merge_overlap_threshold,
                    )
                )
                event_id += 1

    truth = np.concatenate(all_truth)
    prediction = np.concatenate(all_prediction)
    cluster = aggregate_cluster_metrics(events)
    total_seconds = sum(total_per_event) / 1000
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "architecture": type(model).__name__,
            "total_parameters": total_parameters,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "checkpoint_size_mb": args.checkpoint.stat().st_size / 1024**2,
        },
        "dataset": {
            "data_dir": str(args.data_dir.resolve()),
            "events": len(events),
            "nodes": int(truth.size),
        },
        "decoder": {
            "beta_threshold": args.beta_thres,
            "distance_threshold": args.dist_thres,
            "background_id": 0,
            "match_iou_threshold": args.match_iou_threshold,
            "split_merge_overlap_threshold": args.split_merge_overlap_threshold,
        },
        "node_metrics": node_binary_metrics(truth, prediction),
        "cluster_metrics": cluster,
        "compute": {
            "mean_forward_latency_ms": float(np.mean(forward_per_event)),
            "median_forward_latency_ms": float(np.median(forward_per_event)),
            "p95_forward_latency_ms": float(np.percentile(forward_per_event, 95)),
            "mean_total_latency_ms": float(np.mean(total_per_event)),
            "median_total_latency_ms": float(np.median(total_per_event)),
            "p95_total_latency_ms": float(np.percentile(total_per_event, 95)),
            "throughput_events_per_second": (
                len(events) / total_seconds if total_seconds else 0.0
            ),
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
        },
    }
    return metrics, events


def main() -> None:
    """Run evaluation and persist all requested report artifacts."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    metrics, events = evaluate(args)
    write_results(args.output_dir, metrics, events)
    print(f"Baseline evaluation written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
