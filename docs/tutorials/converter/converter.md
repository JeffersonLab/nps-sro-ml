# ROOT-to-PyTorch converter

The C++ converter turns NPS ROOT events into graph data serialized with
LibTorch. Each saved event is a numbered `.pt` file containing:

```text
(node_features, edge_index, edge_attributes, node_targets, node_positions)
```

Nodes represent calorimeter blocks, targets identify truth or reconstructed
clusters, positions contain detector column and row, and edges are optional.
The resulting `.pt` layout is described in the
[dataset tutorial](../pytorch/dataset.md).

## Programs

The CMake build creates three executables in `converter/build/`:

| Executable | Purpose |
| --- | --- |
| `converter.exe` | Convert replayed data using HCANA cluster IDs |
| `reco_vtp.exe` | Emulate fADC250/VTP reconstruction and match VTP clusters |
| `sim_data.exe` | Mix simulated events and convert them to graphs |

All three programs accept one or more ROOT files, select a tree and event
range, and write `00000000.pt`, `00000001.pt`, and so on.

## Build

The converter requires CMake 3.18+, C++17, ROOT, and LibTorch. The supplied
Apptainer/Singularity image provides these dependencies. From the repository
root:

```bash
apptainer build converter/image.sif converter/image.def

REPO="$PWD"
apptainer exec --bind "$REPO:$REPO" converter/image.sif \
  cmake -S "$REPO/converter" -B "$REPO/converter/build"

apptainer exec --bind "$REPO:$REPO" converter/image.sif \
  cmake --build "$REPO/converter/build" --parallel 8
```

Replace `apptainer` with `singularity` on systems using the older command
name. For native compilation and more environment details, see
[Installation](../../installation.md#build-the-c-converter).

Check the compiled command-line interface before a production run:

```bash
converter/build/converter.exe --help
converter/build/reco_vtp.exe --help
converter/build/sim_data.exe --help
```

## Common options

| Option | Meaning |
| --- | --- |
| `-i, --input-files` | One or more input ROOT files |
| `-o, --output-dir` | Directory for numbered `.pt` files |
| `-t, --tree-name` | Input ROOT tree name |
| `-n, --n-events` | Number of entries to inspect; `-1` means all |
| `--start-event` | First tree entry to inspect |
| `--geo-config` | NPS geometry/channel-map CSV |
| `--edge-creation` | Enable graph-edge creation |
| `--edge-algorithm` | `fully_connected` or `center_to_neighbor` |
| `-d, --debug` | Print additional diagnostics |

Without `--edge-creation`, the saved graph has no constructed edges. Input and
output paths must be visible inside the container; bind directories outside
the home directory explicitly.

## HCANA cluster conversion

`converter.exe` reads waveform data and uses HCANA's
`NPS_cal_fly_block_clusterID` labels:

```bash
converter/build/converter.exe \
  --input-files /data/run_4599.root \
  --output-dir /data/run_4599_pt \
  --tree-name T \
  --geo-config database/geo/channel_map.csv \
  --start-event 0 \
  --n-events 1000 \
  --clus-min 0 --clus-max 1000 \
  --sig-min 0 --sig-max 1000 \
  --edge-creation \
  --edge-algorithm fully_connected
```

The program keeps an event only when both counts are strictly inside their
ranges:

```text
clus-min < number of clusters < clus-max
sig-min  < number of active blocks < sig-max
```

Corrupt or empty waveforms are skipped. The final console message reports the
number of inspected and saved events.

## VTP reconstruction

`reco_vtp.exe` needs run-dependent VME and VTP settings. Generate them from
JLab's electronic log while connected to the JLab network:

```bash
uv run python database/jlog/get_run_config.py \
  --run 4599 \
  --channel-map database/geo/channel_map.csv \
  --output-dir database/jlog
```

This writes `nps_run_4599_vme_config.csv` and
`nps_run_4599_vtp_config.csv`. Run the reconstruction with explicit config
paths:

```bash
converter/build/reco_vtp.exe \
  --input-files /data/run_4599.root \
  --output-dir /data/run_4599_vtp_pt \
  --tree-name T \
  --start-event 0 \
  --n-events 1000 \
  --vme-config database/jlog/nps_run_4599_vme_config.csv \
  --vtp-config database/jlog/nps_run_4599_vtp_config.csv \
  --geo-config database/geo/channel_map.csv \
  --energy-diff 5.0 \
  --time-window 15 93 \
  --edge-creation \
  --edge-algorithm fully_connected
```

`--energy-diff` sets the allowed energy difference when matching emulated and
recorded VTP clusters. `--time-window` accepts the two time-window bounds in
nanoseconds.

## Simulated data

`sim_data.exe` combines groups of simulated events before constructing each
graph:

```bash
converter/build/sim_data.exe \
  --input-files /data/simulation.root \
  --output-dir /data/simulation_pt \
  --tree-name nerd \
  --start-event 0 \
  --n-events 1000 \
  --overlaps 5 \
  --geo-config database/geo/channel_map.csv \
  --edge-creation \
  --edge-algorithm fully_connected
```

`--overlaps` controls how many input events are combined. The CLI also accepts
`--dt`, but the current implementation does not apply that value to the
generated pulse times.

## Validate output

Start with a small `--n-events` value and confirm files are produced:

```bash
ls -lh /data/run_4599_pt

PYTHONPATH=pytorch_src uv run python - <<'PY'
import torch

values = torch.load("/data/run_4599_pt/00000000.pt", weights_only=False)
for name, value in zip(
    ("x", "edge_index", "edge_attr", "y", "pos"),
    values,
):
    print(name, tuple(value.shape))
PY
```

The converter creates its output directory automatically. It does not prevent
an existing numbered file from being overwritten, so use a new output
directory for each conversion.

## Related documentation

- [Installation and native build instructions](../../installation.md)
- [PyTorch dataset formats](../pytorch/dataset.md)
- [VTP Manual](/VTP-Manual_1.pdf)
