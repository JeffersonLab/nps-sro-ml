import pathlib
from typing import Any, Optional

import pandas as pd
import torch

from base.model import BaseModel
from datasets.nps import get_node_index_from_position
from utils.graph import pack_to_graph_batches


class ObjectCondensationBaseModel(BaseModel):
    """
    Shared OC model utilities for preparing node features before the forward pass.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.input_type = kwargs.get("input_type", "pulse_set")
        self.vme_config: dict[str, torch.Tensor] = {}
        self.vtp_config: dict[str, torch.Tensor] = {}

    def configure_input_preprocessing(self, config: Optional[dict] = None) -> None:
        args = self._resolve_preprocessing_args(config)
        self.vme_config = self._load_config(
            args.get("vme_config_path", args.get("vme_config"))
        )
        self.vtp_config = self._load_config(
            args.get("vtp_config_path", args.get("vtp_config"))
        )

    def _resolve_preprocessing_args(self, config: Optional[dict]) -> dict:
        if config is None:
            return {}

        trainer_cfg = config.get("trainer")
        if isinstance(trainer_cfg, dict):
            return trainer_cfg.get("args", {})

        return config

    def _load_config(
        self, path: Optional[pathlib.Path | str]
    ) -> dict[str, torch.Tensor]:
        if path is None:
            return {}

        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        df = pd.read_csv(path)
        if "channel" not in df.columns:
            raise ValueError("CSV must contain a 'channel' column")

        config = {}
        for col in df.columns:
            if col == "channel":
                continue
            config[col] = torch.from_numpy(df[col].to_numpy(dtype="float32", copy=False))
        return config

    def preprocess_features(self, data: Any) -> torch.Tensor:
        if self.input_type != "waveform":
            return data.x

        wf = data.x
        pos = data.pos
        channels = get_node_index_from_position(pos[:, 1], pos[:, 0])
        return self._preprocess_wf(wf, channels)

    def _preprocess_wf(self, wf: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        ped = self.vme_config.get("FADC250_ALLCH_PED")
        if ped is None:
            return wf

        if ped.ndim == 1:
            ped = ped.unsqueeze(-1)

        if channels.shape[0] != wf.shape[0]:
            raise ValueError(
                f"Channels tensor length ({channels.shape[0]}) does not match number "
                f"of waveforms ({wf.shape[0]})."
            )

        if channels.numel() > 0 and (
            channels.min() < 0 or channels.max() >= ped.shape[0]
        ):
            raise ValueError(
                f"Waveform channels must be in [0, {ped.shape[0]}), got "
                f"[{channels.min().item()}, {channels.max().item()}]."
            )

        return wf - ped.to(wf.device)[channels.long()]

    def unpack_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_type != "pulse_set":
            return x, torch.ones(x.shape, dtype=torch.bool, device=x.device)

        feature_length = x.shape[-1]
        pulse_count = feature_length // 2
        x_out = x.reshape(*x.shape[:-1], pulse_count, 2)
        mask = x_out.abs().sum(dim=-1) > 0
        return x_out, mask

    def get_batch_vector(
        self, x: torch.Tensor, batch: Optional[torch.Tensor]
    ) -> torch.LongTensor:
        if batch is None:
            return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return batch

    def prepare_graph_inputs(
        self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        x_out, idx_out, node_mask = pack_to_graph_batches(x, [pos], batch=batch)
        x_graph = x_out[0]
        pos_graph = x_out[1]
        x_graph, fea_mask = self.unpack_features(x_graph)
        return x_graph, pos_graph, fea_mask, node_mask, idx_out
