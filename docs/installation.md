# Installation

This repository has two independent environments:

- Python 3.11 or newer for model development, training, and inference.
- C++17, ROOT, and LibTorch for the ROOT-to-PyTorch converters in `converter/`.

Run the commands on this page from the repository root unless noted otherwise.

## Python environment

The project is locked with [uv](https://docs.astral.sh/uv/). If your system does
not already have `uv` installed, see the [uv guide](./tutorials/general/uv.md).

::: tip JLab users
Run the `uv sync` commands below from a terminal inside
[JLab VDI](./tutorials/general/vdi.md), rather than through SSH on ifarm. VDI is
already inside the Jefferson Lab network and is the preferred environment for
installing the Python and PyTorch dependencies.
:::

Clone the repository, and select exactly one PyTorch build:

```bash
git clone https://github.com/JeffersonLab/nps-sro-ml.git
cd nps-sro-ml

# Portable CPU environment
uv sync --extra torch-cpu
```

For an NVIDIA host, replace `torch-cpu` with the CUDA build supported by its
driver:

```bash
uv sync --extra torch-cu126   # CUDA 12.6
uv sync --extra torch-cu128   # CUDA 12.8
uv sync --extra torch-cu130   # CUDA 13.0
```

The PyTorch extras conflict intentionally, so do not select more than one. The
experimental `torch-2-11` extra is also mutually exclusive with them. `uv sync`
creates `.venv/`; commands can then be run with `uv run` without manually
activating it.

Verify the environment:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run pytest pytorch_src/tests
```

To activate the environment in an interactive shell instead:

```bash
source .venv/bin/activate
```

## Build the C++ converter

The converter requires CMake 3.18+, a C++17 compiler, ROOT (`Core`, `RIO`, and
`Tree`), and LibTorch. The supplied Apptainer/Singularity recipe is the most
reproducible route.

### Container build

From the repository root:

```bash
apptainer build converter/image.sif converter/image.def
```

On installations which still use the old executable name, replace `apptainer`
with `singularity`. Building an image may require `--fakeroot` or a remote
builder, depending on site policy.

Build the converter inside the image. Bind the repository explicitly when it is
outside your home directory:

```bash
REPO="$PWD"
apptainer exec --bind "$REPO:$REPO" converter/image.sif \
  cmake -S "$REPO/converter" -B "$REPO/converter/build"

apptainer exec --bind "$REPO:$REPO" converter/image.sif \
  cmake --build "$REPO/converter/build" --parallel 8
```

This creates three programs in `converter/build/`:

| Program | Input/use |
| --- | --- |
| `converter.exe` | Replayed NPS ROOT data with HCANA cluster labels |
| `reco_vtp.exe` | Replayed data reconstructed with VTP/fADC settings |
| `sim_data.exe` | Simulated ROOT events, optionally overlaid |

### Native build

If ROOT is already configured in the shell, the PyTorch wheel can supply
LibTorch and its CMake metadata:

```bash
TORCH_PREFIX="$(uv run python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S converter -B converter/build -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
cmake --build converter/build --parallel 8
```

If CMake cannot find ROOT, source the site's ROOT setup script first or append
its installation prefix to `CMAKE_PREFIX_PATH`. At runtime, ROOT and LibTorch
shared libraries must remain discoverable in `LD_LIBRARY_PATH`.

## Generating training data

Always inspect the installed program's options first:

```bash
converter/build/converter.exe --help
converter/build/reco_vtp.exe --help
converter/build/sim_data.exe --help
```

### HCANA-labelled replay data

```bash
converter/build/converter.exe \
  --input-files /data/run_4599.root \
  --output-dir data/run_4599_pt \
  --tree-name T \
  --geo-config database/geo/channel_map.csv \
  --start-event 0 \
  --n-events 1000 \
  --clus-min 1 --clus-max 1000 \
  --sig-min 1 --sig-max 1000 \
  --edge-creation \
  --edge-algorithm fully_connected
```

`--input-files` accepts one or more paths. `--n-events -1` processes all events.
Without `--edge-creation`, no graph edges are constructed. Output files are
numbered `.pt` files containing node features, edge indices, edge attributes,
node targets, and detector positions.

### VTP reconstruction

First generate the run-dependent VME and VTP CSV files:

```bash
uv run python database/jlog/get_run_config.py \
  --run 4599 \
  --channel-map database/geo/channel_map.csv \
  --output-dir database/jlog
```

Then run the VTP converter:

```bash
converter/build/reco_vtp.exe \
  --input-files /data/run_4599.root \
  --output-dir data/run_4599_vtp_pt \
  --tree-name T \
  --n-events 1000 \
  --vme-config database/jlog/nps_run_4599_vme_config.csv \
  --vtp-config database/jlog/nps_run_4599_vtp_config.csv \
  --geo-config database/geo/channel_map.csv \
  --energy-diff 5.0 \
  --time-window 15 93 \
  --edge-creation \
  --edge-algorithm fully_connected
```

When using a container, wrap the executable command in `apptainer exec` and
bind every input and output directory. Container paths must match the paths
passed to the program.

### Simulated data

```bash
converter/build/sim_data.exe \
  --input-files /data/simulation.root \
  --output-dir data/simulation_pt \
  --tree-name nerd \
  --n-events 1000 \
  --overlaps 5 \
  --dt 32 \
  --geo-config database/geo/channel_map.csv
```

Use absolute paths in batch jobs, check the destination has enough space, and
start with a small `--n-events` value before launching a full conversion.
