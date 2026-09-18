# Training data

The converters transform the [raw ROOT sources](./raw.md) into event-level graph
datasets for PyTorch.

## Locations

| Dataset | Format | JLab path or status |
| --- | --- | --- |
| VTP-reconstructed graphs | One LibTorch `.pt` file per event | `/expphy/volatile/hallc/c-kaonlt/ckin/nps-data/reco_vtp/*.pt` |
| Geant4+SIMC overlap graphs | One LibTorch `.pt` file per combined event | `/lustre24/expphy/volatile/hallc/c-kaonlt/ckin/nps-data/geant4_overlaps/*.pt` |
| VTP/JANA2 event arrays | One directory of `.npy` arrays per event | `users/ckin/nps-data/vtp_cluster_data` in the work area; temporary |

The volatile and work-area datasets are not archival. Confirm that a path still
exists before submitting jobs, and avoid treating it as the only copy.

## Common LibTorch format

The tensor fields follow the graph-data conventions described in PyTorch
Geometric's [Creating Graph Datasets](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)
guide, with node features, connectivity, targets, and positions stored for each
event.

Each `.pt` event serializes five tensors in this order:

```python
(x, edge_index, edge_attr, y, pos)
```

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `x` | `[num_nodes, num_features]` | Waveform or pulse features |
| `edge_index` | `[2, num_edges]` | Optional intra-cluster graph connectivity |
| `edge_attr` | `[num_edges, 0]` | Empty edge-feature tensor in the current converters |
| `y` | `[num_nodes, 1]` | Cluster identifier for each node |
| `pos` | `[num_nodes, 2]` | Detector `(column, row)` |

Block IDs are used while constructing a graph but are not saved as a separate
tensor. See [Datasets and data loaders](../tutorials/pytorch/dataset.md) for
loading both the `.pt` and `.npy` layouts.

## VTP-reconstructed graphs

`reco_vtp.exe` reads the same replay waveforms, emulates the fADC250 and VTP
clustering logic, and matches reconstructed clusters to recorded VTP clusters.
It saves the same five-tensor layout, with waveform features, reconstructed
cluster IDs, detector positions, and optional intra-cluster edges.

## Geant4+SIMC overlap graphs

`sim_data.exe` accumulates `--overlaps` consecutive simulation events and
emits one combined graph:

- **Node:** a calorimeter block occurrence belonging to a reconstructed
  cluster.
- **Node feature `x`:** all pulse `(energy, time)` pairs for that physical
  block, padded with zeros to `4 * overlaps` values. The default
  `overlaps=5` produces 20 features per node.
- **Target `y`:** a graph-local cluster ID starting at `1` across the
  combined input events.
- **Position `pos`:** detector `(column, row)`.
- **Edges:** optional intra-cluster connectivity.

Only complete groups of `overlaps` input events are saved. The current
implementation accepts `--dt` but does not apply that time-gap value when
building pulse features.

## Access and conversion

Use JLab VDI or another host with access to the listed filesystems. To produce a
new dataset, follow the
[ROOT-to-PyTorch converter tutorial](../tutorials/converter/converter.md).
