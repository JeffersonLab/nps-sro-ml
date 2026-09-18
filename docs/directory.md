# Repository structure

The repository is organized by processing stage: C++ converts detector data,
Python trains and evaluates models, and VitePress publishes the documentation.

```text
nps-sro-ml/
├── converter/                 # ROOT-to-PyTorch C++ conversion
│   ├── include/               # C++ headers and bundled argument parser
│   ├── sources/               # Shared converter implementation
│   ├── standalone/
│   │   └── sources/           # converter, VTP, and simulation executables
│   ├── CMakeLists.txt         # C++ library and executable build
│   └── image.def              # Apptainer/Singularity image definition
├── database/
│   ├── geo/                   # NPS geometry and channel map
│   └── jlog/                  # Run-configuration retrieval utilities
├── pytorch_src/
│   ├── base/                  # Base model, trainer, loader, and scaler APIs
│   ├── datasets/              # NPS dataset and graph batching
│   ├── inference/             # Inference managers, metrics, and reports
│   ├── layers/                # Reusable neural-network and graph layers
│   ├── models/                # Model architectures and losses
│   ├── training/              # Training-loop implementations
│   ├── utils/                 # Configuration, graph, and TensorBoard helpers
│   └── tests/                 # Python unit tests
├── scripts/
│   ├── train.py               # Configuration-driven training entry point
│   └── inference.py           # Checkpoint evaluation entry point
├── docs/
│   ├── .vitepress/            # Site configuration and custom theme
│   ├── contributing/          # Contribution guides
│   ├── public/                # Static files copied into the built site
│   └── tutorials/             # General, converter, and PyTorch tutorials
├── .github/workflows/         # Continuous-integration workflows
├── pyproject.toml             # Python metadata and dependency declarations
├── uv.lock                    # Reproducible Python dependency resolution
└── README.md                  # Repository landing page
```

## Generated directories

The following paths are created locally and should not be treated as source:

- `.venv/`: Python environment created by `uv`.
- `converter/build/`: CMake build tree and converter executables.
- `docs/node_modules/`: documentation dependencies installed by npm.
- `docs/.vitepress/dist/`: generated static documentation site.
- Training output directories such as `saved/`, containing checkpoints and
  TensorBoard logs.

See [Installation](./installation.md) for environment setup and
[Tutorials](./tutorials/) for the main workflows.
