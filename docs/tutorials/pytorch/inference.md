# Inference

`scripts/inference.py` reconstructs a model directly from a training checkpoint,
runs a configured inference manager over the validation split, and writes a
report. A checkpoint is self-describing: it contains the model module, class,
constructor metadata, and state dict.

## Write an inference class

Object-condensation inference is split into typed result containers and a
manager. Subclass `BaseOcInferenceManager` and implement four hooks:

| Hook | Responsibility |
| --- | --- |
| `_define_batch(data)` | Return one graph ID per node |
| `_extract_truth(data, batch=...)` | Return fields required by the result dataclass |
| `_prepare_model_inputs(data)` | Apply exactly the same preprocessing as training |
| `_infer_graph(*outputs)` | Convert one graph's model output into predictions |

The manager supplies device handling, `torch.no_grad()`, splitting a minibatch
back into events, and CSV/JSON export. A compact custom implementation is:

```python
import pathlib
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import torch

from inference.oc_inference import (
    BaseOcInferenceManager,
    BaseOcInferenceResults,
    BaseOcInferenceResultsPerGraph,
    oc_inference_per_graph,
)


@dataclass
class MyResultPerGraph(BaseOcInferenceResultsPerGraph):
    truth_ids: torch.Tensor


class MyResults(BaseOcInferenceResults):
    result_type: ClassVar[type[MyResultPerGraph]] = MyResultPerGraph


class MyInferenceManager(BaseOcInferenceManager):
    results_type = MyResults

    def _define_batch(self, data: Any) -> torch.Tensor:
        if hasattr(data, "batch"):
            return data.batch
        return torch.zeros(len(data.x), dtype=torch.long, device=data.x.device)

    def _extract_truth(self, data: Any, *, batch: torch.Tensor) -> Mapping[str, Any]:
        return {"truth_ids": data.y.squeeze(-1).long()}

    def _prepare_model_inputs(self, data: Any) -> tuple[torch.Tensor, ...]:
        # Keep this identical to the trainer's preprocessing.
        from datasets.nps import NCOLS, NROWS, NTIME

        energy = data.x[:, 0]
        scaled_x = torch.stack(
            (
                energy / 1600,
                torch.log1p(energy),
                2 * data.x[:, 1] / NTIME - 1,
            ),
            dim=-1,
        )
        scaled_pos = torch.stack(
            (
                2 * data.pos[:, 0] / NCOLS - 1,
                2 * data.pos[:, 1] / NROWS - 1,
            ),
            dim=-1,
        )
        return scaled_x, scaled_pos

    def _infer_graph(self, *model_outputs: torch.Tensor) -> Mapping[str, Any]:
        x_c, beta = model_outputs[:2]
        object_ids, min_d = oc_inference_per_graph(
            x_c,
            beta,
            beta_thres=self.hyperparameters.beta_thres,
            dist_thres=self.hyperparameters.dist_thres,
            empty_idx=self.hyperparameters.empty_idx,
        )
        return {
            "object_ids": object_ids,
            "x_c": x_c,
            "beta": beta,
            "min_d": min_d,
        }

    def report(self, save_dir: pathlib.Path, **kwargs: Any) -> None:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.export(save_dir / "results.csv")
        self.export(save_dir / "results.json")
```

Place the class in an importable module such as
`pytorch_src/inference/my_inference.py`. If you add result fields, every field
must have one value per node (or be a scalar that can be repeated per node).

For the provided hit model, use
`inference.vtp_hit_inference.VtpHitOcInferenceManager`. Its preprocessing
matches the VTP hit trainer; it performs latent-space clustering, aggregates
trigger logits within each predicted object, calculates metrics, and produces
diagnostic figures.

## Inference configuration

Create `config/inference.json`:

```json
{
  "name": "vtp_hit_inference",
  "save_dir": "saved/inference/vtp_hit_baseline",
  "n_gpu": 1,
  "model_pth": "saved/models/vtp_hit_baseline/RUN_ID/model_best.pth",
  "data_loader": {
    "module": "datasets.nps",
    "type": "NPSDataLoader",
    "args": {
      "data_dir": "/absolute/path/to/npy-events",
      "source": "npy",
      "feature_mode": "hit",
      "batch_size": 16,
      "shuffle": false,
      "validation_split": 0.2,
      "num_workers": 4,
      "use_torch_loader": false
    }
  },
  "inference": {
    "module": "inference.vtp_hit_inference",
    "type": "VtpHitOcInferenceManager",
    "args": {
      "hyperparameters": {
        "beta_thres": 0.5,
        "dist_thres": 0.5,
        "sig_thres": 0.5,
        "q_min": 0.3,
        "empty_idx": -1
      }
    }
  },
  "report": {
    "seed": 42,
    "num_det_plots": 10
  }
}
```

The script calls `split_validation()` and infers on that returned loader. A
non-zero `validation_split` is therefore required. The current loader uses a
fixed split seed of `0`, so the same dataset ordering and split size reproduce
the held-out events. `shuffle: false` makes the inference traversal easier to
follow.

Thresholds have distinct roles:

- `beta_thres`: minimum condensation confidence for a cluster seed.
- `dist_thres`: maximum latent distance for assigning a hit to a seed.
- `sig_thres`: trigger-probability decision threshold.
- `empty_idx`: label used for noise/unassigned hits; it must match training.
- `q_min`: offset in the beta-derived weighting used to aggregate trigger logits.

## Run inference

```bash
uv run python scripts/inference.py -c config/inference.json
```

Override only the checkpoint path from the command line when comparing models:

```bash
uv run python scripts/inference.py \
  -c config/inference.json \
  --model saved/models/vtp_hit_baseline/RUN_ID/model_best.pth
```

The configured `save_dir` receives:

```text
results.csv
results.json
stats_summary.json
metrics.json
figures/
├── trigger_confusion_matrix.png
├── hit_confusion_matrix.png
├── triggered_hit_confusion_matrix.png
├── beta_distribution.png
├── min_distance_distribution.png
├── object_size_distribution.png
├── num_objects_distribution.png
└── event_objects/
```

If `save_dir` is omitted, reports are written beside the checkpoint.

## Programmatic use

For notebooks or threshold scans, load the checkpoint with the same helper used
by the CLI:

```python
from pathlib import Path

from scripts.inference import load_model
from datasets.nps import NPSDataLoader
from inference.vtp_hit_inference import VtpHitOcInferenceManager

model = load_model(Path("saved/models/vtp_hit_baseline/RUN_ID/model_best.pth"))
model = model.to("cuda").eval()

loader = NPSDataLoader(
    data_dir="/absolute/path/to/npy-events",
    source="npy",
    feature_mode="hit",
    batch_size=16,
    validation_split=0.2,
    shuffle=False,
)
manager = VtpHitOcInferenceManager(
    model,
    hyperparameters={
        "beta_thres": 0.5,
        "dist_thres": 0.5,
        "sig_thres": 0.5,
        "empty_idx": -1,
    },
)
manager.infer(loader.split_validation())
manager.report("saved/inference/manual-scan", num_det_plots=10, seed=42)
```

Call `infer` only once per manager instance; repeated calls are skipped to avoid
duplicating results. Create a new manager for each threshold combination.

::: tip Checkpoint portability
Checkpoint loading imports the model's original module and class. Keep custom
model source available under the same module path, and run from this repository
environment. A raw `state_dict` alone is not accepted by the CLI loader.
:::
