# VTP Data


## Replayed ROOT files

The starting point of the project is the replayed ROOT files located at 
```bash
/cache/hallc/c-nps/analysis/pass2/replays/updated/nps_hms_coin_{run}_{seg}_1_-1.root
```
where the files are named according to the run number and segment numbers. The index `1` is kept for historical reason and `-1` means all events are included.

### Important branches

- `NPS.cal.fly.adcSampWaveform` - Raw or pedestal-subtracted fADC250 waveforms for each of the 1080 calorimeter blocks.
- `NPS.cal.vtpClusX, Y, Time` - Row index, column index, and time of the VTP cluster representative.
- `NPS.cal.fly.block_clusterID` - Cluster indices assigned by hcana. Blocks not associated with any cluster (background) are marked with `-1`.


## Torch (`.pt`) files

The machine learning training and test datasets are temporarily stored at
```
/expphy/volatile/hallc/c-kaonlt/ckin/nps-data/reco_vtp/*.pt
```
Each .pt file corresponds to one event, represented as a graph with the following tensor attributes:

### Node-level Tensors

- `nodeIds` - Block indices ranging from `0` to `1079`. Shape : `[num_nodes]`
- `nodeFeatures` - Feature vectors for each block (e.g. waveform samples). Shape : `[num_nodes][num_node_features]`
- `nodeTargets` - Target labels or features. Shape : `[num_nodes][num_targets]`

### Edge-level Tensors

- `edgeIndex` - Graph connectivity (source -> target). Shape : `[2][num_edges]`
- `edgeFeatures` - Features associated with each graph edge [num_edges][num_edge_features]
- `edgeTargetIndex` - Ground-truth graph connectivity (source -> target). Shape : `[2][num_target_edges]`
- `edgeTargetFeatures` - Features associated with the target edges. Shape : `[num_target_edges][num_target_edge_features]`


## Clustering interpretation

For the clustering task, each event is modeled as an undirected graph:
- Nodes correspond to calorimeter blocks.
- Node features are derived from the fADC250 waveforms.
- Truth labels are encoded as target edges, connecting each cluster seed to all its associated blocks and vice versa.

To generate these `.pt` files, see `...` for instruction. To use these `.pt` files for AI/ML application, see `...`.





