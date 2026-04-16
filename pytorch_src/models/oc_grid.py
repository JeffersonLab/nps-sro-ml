import torch
import torch.nn.functional as F
from torch import nn

from base.model import BaseModel
from models.encoders import PulseSetEncoder


def _group_count(channels: int) -> int:
    """Choose a GroupNorm group count that divides `channels`."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TemporalConvBlock(nn.Module):
    """Depthwise-separable residual block for waveform encoding."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.pointwise1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, bias=False)
        self.pointwise2 = nn.Conv1d(2 * channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x * mask
        x = self.depthwise(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.pointwise1(x)
        x = F.gelu(x)
        x = self.pointwise2(x)
        x = self.norm2(x)
        return (residual + x) * mask


class WaveformConvEncoder(BaseModel):
    """
    Lightweight waveform encoder based on temporal convolutions.

    This avoids the O(T^2) memory cost of self-attention over all 110 samples.
    """

    def __init__(self, **kwargs):
        super().__init__()

        hidden = kwargs.get('hidden', 64)
        out_dim = kwargs.get('out_dim', hidden)
        kernel_size = kwargs.get('kernel_size', 5)
        dilations = kwargs.get('dilations', (1, 2, 4))
        downsample_stages = kwargs.get('downsample_stages', 2)

        self.stem = nn.Conv1d(1, hidden, kernel_size=7, padding=3, bias=False)
        self.stem_norm = nn.GroupNorm(_group_count(hidden), hidden)
        self.blocks = nn.ModuleList(
            [
                TemporalConvBlock(
                    channels=hidden,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden,
                    hidden,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
                for _ in range(downsample_stages)
            ]
        )
        self.proj = nn.Conv1d(hidden, out_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected waveform input with shape [B, N, T], got {tuple(x.shape)}"
            )
        if mask.shape != x.shape:
            raise ValueError(
                f"Waveform mask must match x shape, got {tuple(mask.shape)} vs {tuple(x.shape)}"
            )

        batch_size, num_nodes, seq_len = x.shape
        x = x.reshape(batch_size * num_nodes, 1, seq_len)
        mask = mask.reshape(batch_size * num_nodes, 1, seq_len).to(dtype=x.dtype)

        x = self.stem(x * mask)
        x = self.stem_norm(x)
        x = F.gelu(x)

        for idx, block in enumerate(self.blocks):
            x = block(x, mask)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x * mask)
                mask = F.max_pool1d(mask, kernel_size=3, stride=2, padding=1)

        x = self.proj(x) * mask
        denom = mask.sum(dim=-1).clamp_min(1.0)
        x = x.sum(dim=-1) / denom
        return x.reshape(batch_size, num_nodes, -1)


class GridMixerBlock(nn.Module):
    """Local spatial mixing on the calorimeter grid."""

    def __init__(self, channels: int, dilation: int = 1, expansion: int = 2):
        super().__init__()
        padding = dilation
        hidden = channels * expansion

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.pointwise1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.pointwise2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x * mask
        x = self.depthwise(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.pointwise1(x)
        x = F.gelu(x)
        x = self.pointwise2(x)
        x = self.norm2(x)
        return (residual + x) * mask


class GridObjectCondensationModel(BaseModel):
    """
    Memory-efficient object condensation model for the NPS calorimeter.

    Design goals
    ------------
    - Replace per-waveform self-attention with temporal convolutions.
    - Replace global node attention with local 2D mixing on the 36 x 30 detector grid.
    - Keep the trainer-facing interface identical to `models.oc_attn.ObjectCondensationModel`.

    The model accepts sparse node sets as long as `pos[..., 0]` and `pos[..., 1]`
    carry the row and column indices of each valid block. This makes it compatible
    with the current trainer even when background nodes are downsampled.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.pos_dim = kwargs.get('pos_dim', 2)
        self.input_type = kwargs.get('input_type', 'waveform')
        self.grid_rows = kwargs.get('grid_rows', 36)
        self.grid_cols = kwargs.get('grid_cols', 30)

        self.embed_in = kwargs.get('embed_in', 16)
        self.embed_out = kwargs.get('embed_out', 32)
        self.d_model = kwargs.get('d_model', 64)

        self.temporal_hidden = kwargs.get('temporal_hidden', self.d_model)
        self.temporal_kernel_size = kwargs.get('temporal_kernel_size', 5)
        self.temporal_dilations = kwargs.get('temporal_dilations', (1, 2, 4))
        self.temporal_downsample_stages = kwargs.get('temporal_downsample_stages', 2)

        default_grid_dilations = (1, 2, 4, 1, 2, 4)
        self.grid_dilations = kwargs.get('grid_dilations', default_grid_dilations)

        self.oc_mlp_pos_hidden = kwargs.get('oc_mlp_pos_hidden', 64)
        self.oc_mlp_dropout = kwargs.get('oc_mlp_dropout', 0.1)
        self.oc_mlp_beta_hidden = kwargs.get('oc_mlp_beta_hidden', 64)

        if self.input_type == "waveform":
            self.fea_encoder = WaveformConvEncoder(
                hidden=self.temporal_hidden,
                out_dim=self.d_model,
                kernel_size=self.temporal_kernel_size,
                dilations=self.temporal_dilations,
                downsample_stages=self.temporal_downsample_stages,
            )
            self.fea_proj = nn.Identity()
        elif self.input_type == "pulse_set":
            self.fea_encoder = PulseSetEncoder(
                hidden=self.embed_in,
                out_dim=self.embed_out,
            )
            self.fea_proj = (
                nn.Identity()
                if self.embed_out == self.d_model
                else nn.Linear(self.embed_out, self.d_model)
            )
        else:
            raise ValueError(f"Unknown input_type {self.input_type}")

        self.geo_mlp = nn.Sequential(
            nn.Linear(self.pos_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.grid_blocks = nn.ModuleList(
            [
                GridMixerBlock(channels=self.d_model, dilation=dilation)
                for dilation in self.grid_dilations
            ]
        )

        self.head_norm = nn.LayerNorm(self.d_model)
        self.oc_mlp_pos = nn.Sequential(
            nn.Linear(self.d_model, self.oc_mlp_pos_hidden),
            nn.GELU(),
            nn.Dropout(self.oc_mlp_dropout),
            nn.Linear(self.oc_mlp_pos_hidden, self.oc_mlp_pos_hidden),
            nn.GELU(),
            nn.Linear(self.oc_mlp_pos_hidden, self.pos_dim),
        )
        self.oc_mlp_beta = nn.Sequential(
            nn.Linear(self.d_model, self.oc_mlp_beta_hidden),
            nn.GELU(),
            nn.Linear(self.oc_mlp_beta_hidden, 1),
            nn.Sigmoid(),
        )

    def _validate_inputs(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        fea_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> None:
        if pos.ndim != 3:
            raise ValueError(
                f"Expected pos with 3 dims [B, N, pos_dim], got shape {tuple(pos.shape)}"
            )
        if node_mask.ndim != 2:
            raise ValueError(
                f"Expected node_mask with 2 dims [B, N], got shape {tuple(node_mask.shape)}"
            )
        if pos.shape[:2] != node_mask.shape:
            raise ValueError(
                "pos and node_mask must share [B, N] dimensions, got "
                f"{tuple(pos.shape[:2])} vs {tuple(node_mask.shape)}"
            )

        if self.input_type == "waveform":
            if x.ndim != 3:
                raise ValueError(
                    "For input_type='waveform', expected x shape [B, N, T], got "
                    f"{tuple(x.shape)}"
                )
            if fea_mask.ndim != 3:
                raise ValueError(
                    "For input_type='waveform', expected fea_mask shape [B, N, T], got "
                    f"{tuple(fea_mask.shape)}"
                )
            if x.shape != fea_mask.shape:
                raise ValueError(
                    "For input_type='waveform', x and fea_mask must have the same shape, got "
                    f"{tuple(x.shape)} vs {tuple(fea_mask.shape)}"
                )
        else:
            if x.ndim != 4 or x.shape[-1] != 2:
                raise ValueError(
                    "For input_type='pulse_set', expected x shape [B, N, P, 2], got "
                    f"{tuple(x.shape)}"
                )
            if fea_mask.ndim != 3:
                raise ValueError(
                    "For input_type='pulse_set', expected fea_mask shape [B, N, P], got "
                    f"{tuple(fea_mask.shape)}"
                )
            if x.shape[:-1] != fea_mask.shape:
                raise ValueError(
                    "For input_type='pulse_set', x and fea_mask shapes are incompatible: "
                    f"x[:-1]={tuple(x.shape[:-1])}, fea_mask={tuple(fea_mask.shape)}"
                )

        if x.shape[:2] != pos.shape[:2]:
            raise ValueError(
                "x and pos must share [B, N] dimensions, got "
                f"{tuple(x.shape[:2])} vs {tuple(pos.shape[:2])}"
            )

    def _position_indices(
        self, pos: torch.Tensor, node_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_idx = pos.round().long()
        valid = node_mask.to(dtype=torch.bool)
        rows = pos_idx[..., 0]
        cols = pos_idx[..., 1]

        if valid.any():
            row_ok = (rows[valid] >= 0) & (rows[valid] < self.grid_rows)
            col_ok = (cols[valid] >= 0) & (cols[valid] < self.grid_cols)
            if not torch.all(row_ok & col_ok):
                raise ValueError(
                    "Valid node positions must fall inside the detector grid "
                    f"[0, {self.grid_rows}) x [0, {self.grid_cols})."
                )

        return rows, cols

    def _normalize_pos(self, pos: torch.Tensor) -> torch.Tensor:
        scale = pos.new_tensor(
            [
                max(self.grid_rows - 1, 1),
                max(self.grid_cols - 1, 1),
            ]
        )
        return pos / scale

    def _scatter_to_grid(
        self,
        x: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, channels = x.shape
        valid = node_mask.to(dtype=torch.bool)

        grid = x.new_zeros(batch_size, channels, self.grid_rows, self.grid_cols)
        occupancy = x.new_zeros(batch_size, 1, self.grid_rows, self.grid_cols)

        batch_idx = (
            torch.arange(batch_size, device=x.device)
            .unsqueeze(1)
            .expand(batch_size, num_nodes)
        )

        grid[batch_idx[valid], :, rows[valid], cols[valid]] = x[valid]
        occupancy[batch_idx[valid], 0, rows[valid], cols[valid]] = 1.0
        return grid, occupancy

    def _gather_from_grid(
        self,
        grid: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, _, _ = grid.shape
        num_nodes = rows.shape[1]
        valid = node_mask.to(dtype=torch.bool)
        batch_idx = (
            torch.arange(batch_size, device=grid.device)
            .unsqueeze(1)
            .expand(batch_size, num_nodes)
        )

        gathered = grid.new_zeros(batch_size, num_nodes, channels)
        grid = grid.permute(0, 2, 3, 1)
        gathered[valid] = grid[batch_idx[valid], rows[valid], cols[valid]]
        return gathered

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        fea_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(x, pos, fea_mask, node_mask)

        x = self.fea_encoder(x, fea_mask)
        x = self.fea_proj(x)

        pos_idx = pos.round()
        pos_emb = self.geo_mlp(self._normalize_pos(pos_idx))
        node_features = x + pos_emb

        rows, cols = self._position_indices(pos_idx, node_mask)
        grid, occupancy = self._scatter_to_grid(node_features, rows, cols, node_mask)

        for block in self.grid_blocks:
            grid = block(grid, occupancy)

        node_features = self._gather_from_grid(grid, rows, cols, node_mask)
        node_features = self.head_norm(node_features)
        node_features = node_features * node_mask.unsqueeze(-1).to(node_features.dtype)

        x_c = self.oc_mlp_pos(node_features)
        beta = self.oc_mlp_beta(node_features)

        x_c = x_c * node_mask.unsqueeze(-1).to(x_c.dtype)
        beta = beta * node_mask.unsqueeze(-1).to(beta.dtype)
        return x_c, beta
