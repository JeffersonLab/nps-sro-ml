# Training

`scripts/train.py` builds the complete training job from JSON:

```text
JSON config -> data loader -> model -> optimizer -> scheduler -> trainer.train()
```

Run it from the repository root so relative data and output paths are
predictable.

## Write a trainer

Subclass `base.trainer.BaseTrainer` and implement `_train_epoch`. The base class
owns the epoch loop, early stopping, TensorBoard writer, and checkpointing. A
minimal supervised trainer looks like this:

```python
import torch.nn.functional as F

from base.trainer import BaseTrainer


class MyTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer,
        config,
        device,
        dataloader,
        valid_dataloader=None,
        lr_scheduler=None,
        logger=None,
    ):
        super().__init__(model, optimizer, config, logger)
        self.device = device
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0

        for batch_idx, data in enumerate(self.dataloader):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            _, _, trigger_logit = self.model(data.x, data.pos, data.batch)
            target = data.cluster_type.squeeze(-1).float()
            loss = F.binary_cross_entropy_with_logits(
                trigger_logit.squeeze(-1), target
            )
            loss.backward()
            self.optimizer.step()

            step = (epoch - 1) * len(self.dataloader) + batch_idx
            self.writer.set_step(step, mode="train")
            self.writer.add_scalar("loss", loss.item())
            total_loss += loss.item()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Must contain the trainer's configured mnt_metric.
        return {"loss": total_loss / len(self.dataloader)}
```

Put a custom trainer below `pytorch_src/training/`, then reference its module and
class in the config. In a real trainer, also implement a `torch.no_grad()`
validation pass and return `val_loss` if that is the monitored metric. Override
`export_onnx` if best checkpoints should produce usable ONNX files.

For hit-level object condensation, use the provided
`training.vtp_hit_oc_trainer.ObjectCondensationTrainer`. It already performs:

- energy, time, and detector-position preprocessing;
- graph-unique truth IDs and optional background downsampling;
- attractive, repulsive, coward, noise, and trigger-feature losses;
- validation, scalar/histogram logging, scheduling, and ONNX export.

## Prepare the dataset

Follow [Datasets and data loaders](./dataset.md) to select a supported event
format and verify a batch. The configuration below uses hit features from the
`npy` layout because the VTP trainer requires `cluster_type` labels.

## Complete training configuration

Create `config/train.json` (and the `config/` directory if needed):

```json
{
  "name": "vtp_hit_baseline",
  "save_dir": "saved",
  "n_gpu": 1,
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
  },
  "arch": {
    "module": "models.hit_gnn_oc",
    "type": "HitGnnOcModel",
    "args": {
      "d_model": 32,
      "n_gravnet_layers": 2,
      "gravnet_k": 8,
      "oc_mlp_pos_out": 2
    }
  },
  "optimizer": {
    "module": "torch.optim",
    "type": "AdamW",
    "args": {
      "lr": 0.001,
      "weight_decay": 0.0001
    }
  },
  "lr_scheduler": {
    "module": "torch.optim.lr_scheduler",
    "type": "StepLR",
    "args": {
      "step_size": 20,
      "gamma": 0.5
    }
  },
  "trainer": {
    "module": "training.vtp_hit_oc_trainer",
    "type": "ObjectCondensationTrainer",
    "epochs": 100,
    "save_period": 5,
    "mnt_mode": "min",
    "mnt_metric": "val_loss",
    "early_stop": 15,
    "noise_idx": -1,
    "q_min": 0.3,
    "margin": 1.0,
    "attr_scale": 1.0,
    "repul_scale": 1.0,
    "coward_scale": 1.0,
    "noise_scale": 0.1,
    "feat_scale": 1.0,
    "apply_downsample": false,
    "mask_scale": 5.0
  }
}
```

`module` is an importable Python module and `type` is the attribute imported
from it. Constructor parameters belong in `args`, except trainer runtime and
loss settings: those stay directly in the `trainer` object because the entry
point passes that entire object as `config`.

Set `n_gpu` to `0` for CPU, `1` for one GPU, or a larger value for
`torch.nn.DataParallel`. The script uses all requested devices visible through
`CUDA_VISIBLE_DEVICES`.

## Validate, train, and override settings

Start with the debug path. It loads one batch, exports `my_model.onnx`, and asks
ONNX Runtime to inspect the inputs and outputs:

```bash
uv run python scripts/train.py -c config/train.json --debug
```

Then launch training:

```bash
uv run python scripts/train.py -c config/train.json
```

Learning rate and batch size can be overridden without editing JSON:

```bash
uv run python scripts/train.py -c config/train.json --lr 3e-4 --bs 32
```

Each invocation receives a timestamped run ID and writes:

```text
saved/
├── models/vtp_hit_baseline/<run-id>/
│   ├── config.json
│   ├── checkpoint-epoch5.pth
│   ├── model_best.pth
│   └── model_best.onnx
└── logs/vtp_hit_baseline/<run-id>/
    └── events.out.tfevents...
```

The ONNX export is attempted whenever a new best checkpoint is saved. A failed
export is logged but does not discard the PyTorch checkpoint.

## Monitor with TensorBoard

Point TensorBoard at the common log root:

```bash
uv run tensorboard --logdir saved/logs --port 6006
```

Open the URL printed by TensorBoard (normally `http://localhost:6006`). On a
remote machine, tunnel the port:

```bash
ssh -L 6006:localhost:6006 user@training-host
```

The VTP trainer records the component losses, total loss, learning rate, beta
histograms, and model-parameter histograms. Compare runs using the experiment
name and timestamp shown in TensorBoard.

::: warning Resume behavior
`BaseTrainer._resume_checkpoint` is currently not implemented. Checkpoints can
be loaded for inference, but passing a training resume path will not restore an
interrupted optimizer/epoch state until a trainer implements that method.
:::

Next: [load a checkpoint and run inference](./inference.md).
