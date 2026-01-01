

# Converter Program Documentation

The **converter** program extracts information from ROOT files and converts them into **graph-based datasets** stored in `.pt` format for use with **PyTorch** (C++ frontend / LibTorch). It is designed for **fast, reproducible preprocessing** of calorimeter data for machine-learning workflows.

## Overview

* **Input**: Replayed ROOT files (e.g., NPS / VTP outputs)
* **Output**: Graph data serialized as `.pt` files
* **Target Framework**: PyTorch (C++ / LibTorch)
* **Execution Environment**: Singularity container

Each event is represented as a graph:

* **Nodes** → calorimeter blocks (with waveform- or hit-level features)
* **Edges** → connectivity defined by clustering (HCANA or VTP-based)

## Environment & Dependencies

To maximize performance, the converter is built against **LibTorch (PyTorch C++ API)**:
 
* [PyTorch C++](https://docs.pytorch.org/cppdocs)
* [ROOT](https://root.cern/)
* CMake-based build system

All dependencies are encapsulated inside a **Singularity image** to ensure reproducibility across systems (e.g., ifarm).

## Building the Singularity Image

From the `converter/` directory:

```bash
cd converter
singularity build image.sif image.def
```

This produces `image.sif`, which contains ROOT, LibTorch, and all required system libraries.


## Building the Converter

Singularity automatically binds your **home directory**. However, on systems like **ifarm**, the project and data often reside in group or cache directories. Make sure to explicitly bind:

* the project working directory
* any directory containing input ROOT files or output data

### Build Command

```bash
GROUPDIR="/group/c-nps/${work_dir}/nps-sro-ml"
IMAGE="./image.sif"

singularity exec \
  --bind $GROUPDIR \
  --bind $VOLATILE \
  $IMAGE \
  bash -c "
    cmake -S . -B build
    cmake --build build -j 8
  "
```

Replace `work_dir` with the relative path under `/group/c-nps/` pointing to the project root.

The executables will be created in:

```text
converter/build/*.exe
```

### Interactive Usage

For debugging or exploratory work, you can launch an interactive shell inside the container:

```bash
singularity run image.sif
```

## Running the Converter

Graph construction consists of two conceptual steps:

1. **Node definition**: calorimeter blocks (energy, timing, waveform-derived features)
2. **Edge definition**: block connectivity defined by clusters

Currently, two cluster definitions are supported:

* **HCANA clusters**
  Uses HCANA-reconstructed clusters from the ROOT branch:
  `NPS_cal_fly_block_clusterID`

* **VTP clusters**
  Reconstructs clusters by emulating the **fADC250** response from raw waveforms using VTP logic

### HCANA Clusters

Example command:

```bash

GROUPDIR="/group/c-nps/${work_dir}/nps-sro-ml"
DATADIR="/cache/..."
IMAGE="./image.sif"

singularity exec \
  --bind $GROUPDIR \
  --bind $DATADIR \
  $IMAGE \
  bash -c "
    ./build/converter.exe \
      -i input_file1 input_file2 \
      -o output_dir \
      --n-events 1000 \
      --start-event 0 \
      --tree-name T \
      --clus-min 35 \
      --clus-max 43 \
      --sig-min 10 \
      --sig-max 200
  "
```

#### Command-Line Options

| Option          | Description                     |
| --------------- | ------------------------------- |
| `-i, --input-files`  | Input ROOT file(s) or file list |
| `-o, --output-dir` | Output Directory              |  
| `--n-events`    | Number of events to process     |
| `--start-event` | First event index               |
| `--tree-name`   | ROOT TTree name (e.g., `T`)     |
| `--clus-min`  | minimum number of clusters to consider event |
| `--clus-max`  | maximum number of clusters to consider event |
| `--sig-min`  | minimum number of signals to consider event |
| `--sig-max`  | maximum number of signals to consider event |

Run the following the command for detailed options
```bash
./build/converter.exe --help
```

### VTP clusters
### Preparing Configuration Files

The VTP reconstruction requires **run-dependent configuration files** for:

* VME modules
* VTP modules
* NPS geometry / channel map

To generate these CSV files, navigate to `database/jlog` and run:

```bash
./get_run_config.py \
  --run 4599 \
  --channel-map ../geo/channel_map.csv \
  --output-dir .
```

This produces:

* `nps_run_4599_vtp_config.csv`
* `nps_run_4599_vme_config.csv`

Each file contains per-channel configuration parameters parsed from the database.

#### Reconstruction program

The execution pattern is similar to the HCANA-based converter:

```bash
singularity exec \
  --bind $GROUPDIR \
  --bind $DATADIR \
  $IMAGE \
  bash -c "
    ./build/recoVtp.exe \
      -i input_files \
      --n-events 1000 \
      --start-event 0 \
      --tree-name T \
      --vme-config   database_dir/jlog/nps_run_${run}_vme_config.csv \
      --vtp-config   database_dir/jlog/nps_run_${run}_vtp_config.csv \
      --geo-config   database_dir/geo/channel_map.csv
  "
```

#### Command-Line Options

| Option          | Description                     |
| --------------- | ------------------------------- |
| `--vme-config`  | Path to VME configuration CSV   |
| `--vtp-config`  | Path to VTP configuration CSV   |
| `--geo-config`  | Geometry / channel map CSV      |


## Related Documentation

* [VTP Manual]()

