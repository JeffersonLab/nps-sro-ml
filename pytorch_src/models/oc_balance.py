import torch
import torch.nn.functional as F
from torch import nn

from models.oc_base import ObjectCondensationBaseModel
from models.encoders import PulseSetEncoder, WaveformEncoder


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualMLPBlock(nn.Module):
    """Cheap per-node feature mixing before spatial reasoning."""

    def __init__(self, d_model: int, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * hidden_mult
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class GridConvBlock(nn.Module):
    """Local spatial mixing with depthwise-separable convolutions."""

    def __init__(self, channels: int, dilation: int = 1, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.pointwise2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x * mask)
        x = self.depthwise(x)
        x = F.gelu(x)
        x = self.pointwise1(x)
        x = F.gelu(x)
        x = self.pointwise2(x)
        x = self.norm2(x)
        return (residual + x) * mask


class LatentCrossBlock(nn.Module):
    """
    Low-rank global mixing through a small learned latent set.

    Complexity is O(B * N * M) instead of O(B * N^2), where M is the number of
    latent tokens.
    """

    def __init__(self, d_model: int, num_heads: int, num_latents: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_model))
        self.null_node = nn.Parameter(torch.randn(1, 1, d_model))
        self.latent_q = nn.Linear(d_model, d_model)
        self.latent_k = nn.Linear(d_model, d_model)
        self.latent_v = nn.Linear(d_model, d_model)
        self.latent_out = nn.Linear(d_model, d_model)
        self.node_q = nn.Linear(d_model, d_model)
        self.node_k = nn.Linear(d_model, d_model)
        self.node_v = nn.Linear(d_model, d_model)
        self.node_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.node_mlp = ResidualMLPBlock(
            d_model=d_model, hidden_mult=2, dropout=dropout
        )
        self.latent_mlp = ResidualMLPBlock(
            d_model=d_model, hidden_mult=2, dropout=dropout
        )
        self.node_norm = nn.LayerNorm(d_model)
        self.latent_norm = nn.LayerNorm(d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

    def _cross_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        q = self._split_heads(q_proj(query))
        k = self._split_heads(k_proj(key))
        v = self._split_heads(v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim**0.5)
        scores = scores.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(1),
            torch.finfo(scores.dtype).min,
        )
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        return out_proj(out)

    def forward(self, nodes: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        node_mask = node_mask.to(dtype=torch.bool)
        node_mask_f = node_mask.unsqueeze(-1).to(nodes.dtype)
        null_node = self.null_node.expand(nodes.shape[0], -1, -1)
        nodes_with_null = torch.cat([nodes * node_mask_f, null_node], dim=1)
        key_padding_mask = torch.cat(
            [
                ~node_mask,
                torch.zeros(
                    (node_mask.shape[0], 1),
                    dtype=torch.bool,
                    device=node_mask.device,
                ),
            ],
            dim=1,
        )
        latents = self.latents.expand(nodes.shape[0], -1, -1)

        latents_update = self._cross_attention(
            query=self.latent_norm(latents),
            key=self.node_norm(nodes_with_null),
            value=self.node_norm(nodes_with_null),
            key_padding_mask=key_padding_mask,
            q_proj=self.latent_q,
            k_proj=self.latent_k,
            v_proj=self.latent_v,
            out_proj=self.latent_out,
        )
        latents = latents + latents_update
        latents = self.latent_mlp(latents)

        nodes_update = self._cross_attention(
            query=self.node_norm(nodes),
            key=self.latent_norm(latents),
            value=self.latent_norm(latents),
            key_padding_mask=torch.zeros(
                (latents.shape[0], latents.shape[1]),
                dtype=torch.bool,
                device=latents.device,
            ),
            q_proj=self.node_q,
            k_proj=self.node_k,
            v_proj=self.node_v,
            out_proj=self.node_out,
        )
        nodes = nodes + nodes_update
        nodes = self.node_mlp(nodes)
        return nodes * node_mask_f


class BalancedObjectCondensationModel(ObjectCondensationBaseModel):
    """
    Balanced OC model for high-level calorimeter features.

    Suggested scratch design
    ------------------------
    1. Encode per-block high-level features with a compact shared encoder.
    2. Add explicit detector geometry via learned row/column embeddings.
    3. Mix mostly with local 2D grid convolutions on the 36 x 30 calorimeter.
    4. Recover long-range event context using a tiny latent bottleneck instead of
       full all-to-all attention.

    This is usually a better memory/accuracy tradeoff than global transformer
    blocks when the inputs are already distilled to energy/time-style features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pos_dim = kwargs.get('pos_dim', 2)
        self.grid_rows = kwargs.get('grid_rows', 36)
        self.grid_cols = kwargs.get('grid_cols', 30)

        self.embed_in = kwargs.get('embed_in', 16)
        self.embed_out = kwargs.get('embed_out', 32)
        self.d_model = kwargs.get('d_model', 64)

        self.wf_enc_d_model = kwargs.get('wf_d_model', 32)
        self.wf_enc_layers = kwargs.get('wf_enc_layers', 2)
        self.wf_enc_heads = kwargs.get('wf_enc_heads', 4)
        self.wf_enc_dropout = kwargs.get('wf_enc_dropout', 0.1)
        self.wf_enc_ff = kwargs.get('wf_enc_ff', 4)

        self.feature_layers = kwargs.get('feature_layers', 2)
        self.feature_dropout = kwargs.get('feature_dropout', 0.1)
        self.grid_dilations = kwargs.get('grid_dilations', (1, 2, 1, 3))
        self.num_global_layers = kwargs.get('num_global_layers', 2)
        self.num_latents = kwargs.get('num_latents', 8)
        self.num_heads = kwargs.get('num_heads', 4)
        self.attn_dropout = kwargs.get('attn_dropout', 0.1)

        self.oc_mlp_pos_hidden = kwargs.get('oc_mlp_pos_hidden', 64)
        self.oc_mlp_dropout = kwargs.get('oc_mlp_dropout', 0.1)
        self.oc_mlp_beta_hidden = kwargs.get('oc_mlp_beta_hidden', 64)

        if self.input_type == "pulse_set":
            self.fea_encoder = PulseSetEncoder(
                hidden=self.embed_in,
                out_dim=self.embed_out,
            )
            self.fea_proj = (
                nn.Identity()
                if self.embed_out == self.d_model
                else nn.Linear(self.embed_out, self.d_model)
            )
        elif self.input_type == "waveform":
            self.fea_encoder = WaveformEncoder(
                d_model=self.wf_enc_d_model,
                n_layers=self.wf_enc_layers,
                n_heads=self.wf_enc_heads,
                dropout=self.wf_enc_dropout,
                ff_mult=self.wf_enc_ff,
                out_dim=self.d_model,
            )
            self.fea_proj = nn.Identity()
        else:
            raise ValueError(f"Unknown input_type {self.input_type}")

        self.feature_mixer = nn.ModuleList(
            [
                ResidualMLPBlock(
                    d_model=self.d_model,
                    hidden_mult=2,
                    dropout=self.feature_dropout,
                )
                for _ in range(self.feature_layers)
            ]
        )

        self.row_embed = nn.Embedding(self.grid_rows, self.d_model)
        self.col_embed = nn.Embedding(self.grid_cols, self.d_model)
        self.pos_mlp = nn.Sequential(
            nn.Linear(self.pos_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.grid_blocks = nn.ModuleList(
            [
                GridConvBlock(channels=self.d_model, dilation=dilation)
                for dilation in self.grid_dilations
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                LatentCrossBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    num_latents=self.num_latents,
                    dropout=self.attn_dropout,
                )
                for _ in range(self.num_global_layers)
            ]
        )

        self.out_norm = nn.LayerNorm(self.d_model)
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

    def _normalize_pos(self, pos: torch.Tensor) -> torch.Tensor:
        scale = pos.new_tensor(
            [
                max(self.grid_cols - 1, 1),
                max(self.grid_rows - 1, 1),
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
        rows = rows.clamp(0, self.grid_rows - 1)
        cols = cols.clamp(0, self.grid_cols - 1)
        row_hot = F.one_hot(rows, num_classes=self.grid_rows).to(x.dtype)
        col_hot = F.one_hot(cols, num_classes=self.grid_cols).to(x.dtype)
        spatial = row_hot.unsqueeze(-1) * col_hot.unsqueeze(-2)
        spatial = spatial * node_mask.unsqueeze(-1).unsqueeze(-1).to(x.dtype)

        grid = torch.einsum("bnrc,bnd->bdrc", spatial, x)
        occupancy = spatial.sum(dim=1, keepdim=True)
        return grid, occupancy

    def _gather_from_grid(
        self,
        grid: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        rows = rows.clamp(0, self.grid_rows - 1)
        cols = cols.clamp(0, self.grid_cols - 1)
        row_hot = F.one_hot(rows, num_classes=self.grid_rows).to(grid.dtype)
        col_hot = F.one_hot(cols, num_classes=self.grid_cols).to(grid.dtype)
        spatial = row_hot.unsqueeze(-1) * col_hot.unsqueeze(-2)
        gathered = torch.einsum("bnrc,bdrc->bnd", spatial, grid)
        return gathered * node_mask.unsqueeze(-1).to(gathered.dtype)

    def _geometry_embedding(
        self,
        pos: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.pos_mlp(self._normalize_pos(pos))
            + self.row_embed(rows)
            + self.col_embed(cols)
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        fea_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        cols = pos[..., 0].round().long()
        rows = pos[..., 1].round().long()

        x = self.fea_encoder(x, fea_mask)
        x = self.fea_proj(x)
        for block in self.feature_mixer:
            x = block(x)

        x = x + self._geometry_embedding(pos, rows, cols)
        x = x * node_mask.unsqueeze(-1).to(x.dtype)

        grid, occupancy = self._scatter_to_grid(x, rows, cols, node_mask)
        for block in self.grid_blocks:
            grid = block(grid, occupancy)

        x = self._gather_from_grid(grid, rows, cols, node_mask)
        for block in self.global_blocks:
            x = block(x, node_mask)

        x = self.out_norm(x)
        x = x * node_mask.unsqueeze(-1).to(x.dtype)

        x_c = self.oc_mlp_pos(x)
        beta = self.oc_mlp_beta(x)

        x_c = x_c * node_mask.unsqueeze(-1).to(x_c.dtype)
        beta = beta * node_mask.unsqueeze(-1).to(beta.dtype)
        return x_c, beta
