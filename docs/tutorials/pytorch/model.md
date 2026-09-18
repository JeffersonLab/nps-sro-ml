# Writing a PyTorch model

Models are ordinary `torch.nn.Module` classes loaded dynamically from a JSON
configuration. Project models inherit `base.model.BaseModel`, which adds a
useful parameter count to their string representation and requires `forward`.

Before implementing a model, read [Datasets and data loaders](./dataset.md) for
the event formats, graph-batch fields, validation split, and preprocessing
contract. The example below assumes the VTP hit trainer, which calls the model
with prepared `(x, pos, batch)` tensors.

## Required output contract

The existing object-condensation trainer and inferencer expect:

```python
x_c, beta, trigger_logit = model(x, pos, batch)
```

- `x_c`: `[N, C]` latent condensation coordinates.
- `beta`: `[N, 1]` condensation confidence in `[0, 1]`.
- `trigger_logit`: `[N, 1]` unnormalized binary-classification score.

Keep logits unscaled: the trainer applies binary cross entropy with logits and
the inferencer applies `sigmoid`. `batch` must prevent information leaking
between events in graph or attention operations.

## Minimal compatible model

Create `pytorch_src/models/simple_hit_oc.py`:

```python
import torch
from torch import nn

from base.model import BaseModel


class SimpleHitOcModel(BaseModel):
    """Small baseline implementing the hit-level OC output contract."""

    def __init__(self, hidden_dim: int = 64, latent_dim: int = 2):
        super().__init__()
        # Store constructor values: checkpoint loading uses these as metadata.
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),  # 3 hit features + 2 coordinates
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coordinate_head = nn.Linear(hidden_dim, latent_dim)
        self.beta_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.trigger_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del batch  # This baseline processes nodes independently.
        features = self.encoder(torch.cat((x, pos), dim=-1))
        return (
            self.coordinate_head(features),
            self.beta_head(features),
            self.trigger_head(features),
        )
```

The model is deliberately simple. Production implementations such as
`models.hit_gnn_oc.HitGnnOcModel` use GravNet to exchange information between
nodes. When adding message passing, use `batch` (or already batch-safe edges)
to isolate graphs.

## Make the model configurable

The configuration loader imports a class by full module path; no registry edit
is required:

```json
"arch": {
  "module": "models.simple_hit_oc",
  "type": "SimpleHitOcModel",
  "args": {
    "hidden_dim": 64,
    "latent_dim": 2
  }
}
```

The training checkpoint records the model's module, class name, state dict, and
non-module public attributes. Store every value needed to reconstruct the
architecture as a plain public attribute, as in the example. Avoid deriving
layer sizes from data seen only during `forward`.

## Smoke-test the contract

From the repository root:

```bash
PYTHONPATH=pytorch_src uv run python - <<'PY'
import torch
from models.simple_hit_oc import SimpleHitOcModel

model = SimpleHitOcModel(hidden_dim=32, latent_dim=2)
x = torch.randn(12, 3)
pos = torch.randn(12, 2)
batch = torch.tensor([0] * 5 + [1] * 7)
x_c, beta, trigger_logit = model(x, pos, batch)
assert x_c.shape == (12, 2)
assert beta.shape == trigger_logit.shape == (12, 1)
assert torch.all((0 <= beta) & (beta <= 1))
print(model)
PY
```

After the eager forward pass works, use training's `--debug` mode to exercise
the real dataloader and ONNX export:

```bash
uv run python scripts/train.py -c config/train.json --debug
```

ONNX export is implemented by the selected trainer, so a custom forward
signature must be matched by a custom `export_onnx` implementation.

Next: [write a trainer and run training](./training.md).
