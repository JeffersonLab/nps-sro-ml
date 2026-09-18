# PyTorch workflow

The PyTorch code under `pytorch_src/` provides reusable datasets, graph layers,
models, trainers, and inference managers. Configuration-driven entry points in
`scripts/` assemble these components into a complete experiment.

## Recommended procedure

1. **Set up Python.** Follow the [installation guide](../../installation.md)
   and verify that PyTorch can see the intended CPU or GPU device.
2. **Prepare and inspect data.** Read
   [Datasets and data loaders](./dataset.md) to choose an on-disk format, create
   train/validation loaders, and understand the graph-batch fields.
3. **Select or implement a model.** Follow
   [Writing a model](./model.md) for the input/output contract, configuration,
   checkpoint metadata, and a forward-pass smoke test.
4. **Configure and run training.** Follow [Training](./training.md) to write a
   trainer, create the JSON configuration, run debug mode, save checkpoints,
   and monitor TensorBoard.
5. **Evaluate a checkpoint.** Follow [Inference](./inference.md) to implement
   an inference manager, tune clustering thresholds, and generate metrics and
   diagnostic plots.

In short:

```text
converted data -> dataset/loader -> model -> trainer -> checkpoint -> inference/report
```

## Main components

| Component | Location | Purpose |
| --- | --- | --- |
| Dataset and loader | `pytorch_src/datasets/nps.py` | Read event files, collate graphs, and split validation data |
| Models | `pytorch_src/models/` | Map graph features to object-condensation predictions |
| Trainers | `pytorch_src/training/` | Optimize models, validate, log, checkpoint, and export ONNX |
| Inference | `pytorch_src/inference/` | Restore checkpoints, cluster predictions, calculate metrics, and report |
| Configuration | `pytorch_src/utils/config.py` | Dynamically construct components from JSON |
| Training entry point | `scripts/train.py` | Run a configured training experiment |
| Inference entry point | `scripts/inference.py` | Evaluate a configured checkpoint |

The tutorials use the hit-level object-condensation pipeline as the concrete
example, but the same component boundaries support custom models, trainers,
and inference logic.

Use the [implementation checklist](./summary.md) before starting a full
training run or handing a model off for deployment.
