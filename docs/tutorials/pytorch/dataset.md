# Datasets and data loaders

The PyTorch workflow represents each detector event as a graph.
`datasets.nps.NPSDataset` reads individual events, while
`datasets.nps.NPSDataLoader` combines them into minibatches and optionally
creates a validation split.

## Supported data layouts

| Source | On-disk layout | Typical use |
| --- | --- | --- |
| `npy` | One directory per event containing NumPy arrays | Waveform- or hit-level training, including trigger labels |
| `torch` | One numbered `.pt` file per event | Graph data produced by the C++ converter |

An `npy` event directory must contain:

```text
event_000001/
├── waveforms.npy
├── hits.npy
├── geometry.npy
├── edge_index.npy
├── cluster_index.npy
└── cluster_type.npy
```

Set `feature_mode` to `"waveform"` to use `waveforms.npy`, or to `"hit"` to
use `hits.npy`. The hit-level VTP trainer requires the latter.

A `torch` event stores `(x, edge_index, edge_attr, y, pos)`. This format does
not contain `cluster_type`, so it cannot be used unchanged with the VTP hit
trainer's trigger-classification loss.

## Event and batch fields

The loader yields a PyG `Batch` when `torch_geometric` is available, or the
repository's torch-only `TorchGraphBatch` otherwise.

| Field | Shape | Meaning |
| --- | --- | --- |
| `x` | `[N, F]` | Node features concatenated across all events |
| `pos` | `[N, 2]` | Detector column and row coordinates |
| `edge_index` | `[2, E]` | Directed graph edges |
| `edge_attr` | `[E, A]` or `None` | Optional edge features |
| `y` | `[N]` or `[N, 1]` | Truth cluster ID; noise is normally `-1` |
| `cluster_type` | `[N]` | Trigger label available for the `npy` source |
| `batch` | `[N]` | Event index for every node in the minibatch |

Here, `N` is the total number of nodes across the minibatch. Graph-aware
models must use `batch` or batch-safe edges so information cannot pass between
different events.

## Create a loader

```python
from datasets.nps import NPSDataLoader

loader = NPSDataLoader(
    data_dir="/absolute/path/to/npy-events",
    source="npy",
    feature_mode="hit",
    batch_size=16,
    shuffle=True,
    validation_split=0.2,
    num_workers=4,
)
validation_loader = loader.split_validation()
```

Run standalone examples with `pytorch_src` on the import path:

```bash
PYTHONPATH=pytorch_src uv run python example.py
```

`validation_split` may be a fraction between `0` and `1`, or an integer number
of events. A value of `0` disables validation. The current loader uses a fixed
split seed of `0`, so the same ordered dataset and split size reproduce the
same subsets. Set `use_torch_loader=True` to force torch-only collation even
when PyG is installed.

## Configure the loader

The training and inference entry points construct the loader from JSON:

```json
"data_loader": {
  "module": "datasets.nps",
  "type": "NPSDataLoader",
  "args": {
    "data_dir": "/absolute/path/to/npy-events",
    "source": "npy",
    "feature_mode": "hit",
    "batch_size": 16,
    "shuffle": true,
    "validation_split": 0.2,
    "num_workers": 4,
    "use_torch_loader": false
  }
}
```

Use an absolute `data_dir` in batch jobs. Start with `num_workers: 0` when
debugging; increase it only after confirming the dataset loads correctly.

## Inspect a batch

```python
batch = next(iter(loader))
print("features:", batch.x.shape)
print("positions:", batch.pos.shape)
print("edges:", batch.edge_index.shape)
print("graphs:", batch.batch.unique().numel())
print("cluster IDs:", batch.y.unique())
```

The VTP hit trainer converts raw energy and time into scaled energy,
`log1p(energy)`, and scaled time. It also scales detector coordinates before
calling the model. Inference must apply the same transformation.

Next: [write a model](./model.md) that accepts the prepared graph batch.
