import torch
import torch.nn.functional as F
from torch import nn

from layers.attention import AttentionLayer, FullAttention
from layers.encoders import Encoder, VanillaEncoderLayer
from layers.embed import PositionalEmbedding
from models.oc_base import ObjectCondensationBaseModel


class ResidualMLP(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * hidden_mult
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class MultiPulseWaveformTokenizer(nn.Module):
    """Shared waveform encoder followed by learned pulse-query extraction."""

    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        ff_mult: int = 4,
        num_tokens: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens

        self.sample_embed = nn.Linear(1, d_model)
        self.pos_embed = PositionalEmbedding(d_model=d_model)
        self.layers = nn.ModuleList(
            [
                VanillaEncoderLayer(
                    attn_layer=AttentionLayer(
                        FullAttention(
                            mask_flag=True,
                            attention_dropout=dropout,
                            scale=None,
                        ),
                        d_model=d_model,
                        n_heads=n_heads,
                    ),
                    d_model=d_model,
                    ff_kwargs={"d_ff": d_model * ff_mult, "activation": nn.GELU},
                    dropout=dropout,
                    batchnorm=False,
                )
                for _ in range(n_layers)
            ]
        )
        self.encoder = Encoder(self.layers)
        self.block_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, d_model))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mlp = ResidualMLP(d_model=d_model, hidden_mult=2, dropout=dropout)
        self.time_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, num_samples = x.shape
        x_flat = x.reshape(-1, num_samples, 1)
        mask_flat = mask.reshape(-1, num_samples)

        seq = self.sample_embed(x_flat)
        seq = self.pos_embed(seq)

        attn_mask = (~mask_flat).unsqueeze(1).unsqueeze(1)
        seq, _ = self.encoder(seq, attn_mask=attn_mask)

        valid = mask_flat.unsqueeze(-1).to(seq.dtype)
        pooled = (seq * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        pooled = self.block_pool(pooled)

        queries = self.token_queries.expand(seq.shape[0], -1, -1)
        queries = queries + pooled.unsqueeze(1)
        tokens, _ = self.cross_attn(
            query=queries,
            key=seq,
            value=seq,
            key_padding_mask=~mask_flat,
            need_weights=False,
        )
        tokens = self.token_mlp(self.token_norm(tokens))

        token_time = self.time_head(tokens).squeeze(-1)
        token_mask = mask.any(dim=-1, keepdim=True).expand(-1, -1, self.num_tokens)
        token_mask = token_mask.reshape(batch_size, num_nodes, self.num_tokens)

        tokens = tokens.reshape(batch_size, num_nodes, self.num_tokens, self.d_model)
        block_feature = pooled.reshape(batch_size, num_nodes, self.d_model)
        token_time = token_time.reshape(batch_size, num_nodes, self.num_tokens)
        return tokens, block_feature, token_time, token_mask


class MultiPulseSetTokenizer(nn.Module):
    """Pulse-set fallback path that keeps K pulses as proposal tokens."""

    def __init__(self, d_model: int = 64, num_tokens: int = 2):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.block_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pulses = x.shape[2]
        pad = max(self.num_tokens - num_pulses, 0)
        x_pad = F.pad(x, (0, 0, 0, pad))
        mask_pad = F.pad(mask, (0, pad))

        x_sel = x_pad[:, :, : self.num_tokens]
        mask_sel = mask_pad[:, :, : self.num_tokens]
        tokens = self.norm(self.embed(x_sel))
        tokens = tokens * mask_sel.unsqueeze(-1).to(tokens.dtype)

        all_tokens = self.norm(self.embed(x_pad))
        all_tokens = all_tokens * mask_pad.unsqueeze(-1).to(all_tokens.dtype)
        all_valid = mask_pad.unsqueeze(-1).to(all_tokens.dtype)
        block_feature = (all_tokens * all_valid).sum(dim=2) / all_valid.sum(dim=2).clamp_min(1.0)
        block_feature = self.block_pool(block_feature)

        token_time = x_sel[..., 1]
        return tokens, block_feature, token_time, mask_sel


class SpatioTemporalTransformerBlock(nn.Module):
    """Transformer block with additive spatial-temporal attention bias."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.bias_mlp = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_heads),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = ResidualMLP(d_model=d_model, hidden_mult=2, dropout=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        token_pos: torch.Tensor,
        token_time: torch.Tensor,
        token_mask: torch.Tensor,
        token_gate: torch.Tensor,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(self.norm1(x)))
        k = self._split_heads(self.k_proj(self.norm1(x)))
        v = self._split_heads(self.v_proj(self.norm1(x)))

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        rel_pos = token_pos.unsqueeze(2) - token_pos.unsqueeze(1)
        rel_time = token_time.unsqueeze(2) - token_time.unsqueeze(1)
        bias_inputs = torch.cat(
            [
                rel_pos[..., :1].abs(),
                rel_pos[..., 1:2].abs(),
                rel_time.abs().unsqueeze(-1),
                rel_pos.norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        bias = self.bias_mlp(bias_inputs).permute(0, 3, 1, 2)

        gate_bias = torch.log(token_gate.clamp_min(1e-6)).unsqueeze(1).unsqueeze(2)
        scores = scores + bias + gate_bias

        key_mask = (~token_mask).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(key_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        out = x + self.out_proj(out)
        out = out * token_mask.unsqueeze(-1).to(out.dtype)
        out = self.ff(self.norm2(out))
        return out * token_mask.unsqueeze(-1).to(out.dtype)


class MultiPulseObjectCondensationModel(ObjectCondensationBaseModel):
    """
    Multi-pulse OC model with explicit proposal scores before clustering.

    `forward` keeps the legacy `(x_c, beta)` return for trainer compatibility.
    Full pulse-token outputs are exposed through cached tensors and the
    `propose_pulses` / `cluster_pulses` helpers.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pos_dim = kwargs.get("pos_dim", 2)
        self.d_model = kwargs.get("d_model", 64)
        self.num_pulse_tokens = kwargs.get(
            "num_pulse_tokens",
            kwargs.get("max_objects_per_block", 2),
        )
        self.num_cluster_layers = kwargs.get("n_enc_layers", 2)
        self.num_heads = kwargs.get("num_heads", 4)
        self.attn_dropout = kwargs.get("attn_dropout", 0.1)
        self.oc_mlp_pos_hidden = kwargs.get("oc_mlp_pos_hidden", 64)
        self.oc_mlp_dropout = kwargs.get("oc_mlp_dropout", 0.1)
        self.oc_mlp_beta_hidden = kwargs.get("oc_mlp_beta_hidden", 64)

        self.wf_enc_d_model = kwargs.get("wf_d_model", self.d_model)
        self.wf_enc_layers = kwargs.get("wf_enc_layers", 2)
        self.wf_enc_heads = kwargs.get("wf_enc_heads", 4)
        self.wf_enc_dropout = kwargs.get("wf_enc_dropout", 0.1)
        self.wf_enc_ff = kwargs.get("wf_enc_ff", 4)

        if self.input_type == "waveform":
            self.tokenizer = MultiPulseWaveformTokenizer(
                d_model=self.wf_enc_d_model,
                n_layers=self.wf_enc_layers,
                n_heads=self.wf_enc_heads,
                dropout=self.wf_enc_dropout,
                ff_mult=self.wf_enc_ff,
                num_tokens=self.num_pulse_tokens,
            )
            self.token_proj = (
                nn.Identity()
                if self.wf_enc_d_model == self.d_model
                else nn.Linear(self.wf_enc_d_model, self.d_model)
            )
        elif self.input_type == "pulse_set":
            self.tokenizer = MultiPulseSetTokenizer(
                d_model=self.d_model,
                num_tokens=self.num_pulse_tokens,
            )
            self.token_proj = nn.Identity()
        else:
            raise ValueError(f"Unknown input_type {self.input_type}")

        self.geo_mlp = nn.Sequential(
            nn.Linear(self.pos_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.amp_embed = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.slot_embed = nn.Embedding(self.num_pulse_tokens, self.d_model)
        self.proposal_fuse = ResidualMLP(
            d_model=self.d_model,
            hidden_mult=2,
            dropout=self.attn_dropout,
        )

        self.proposal_score_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid(),
        )
        self.proposal_time_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid(),
        )
        self.proposal_width_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
            nn.Softplus(),
        )
        self.proposal_amplitude_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
            nn.Softplus(),
        )

        self.cluster_blocks = nn.ModuleList(
            [
                SpatioTemporalTransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    dropout=self.attn_dropout,
                )
                for _ in range(self.num_cluster_layers)
            ]
        )
        self.cluster_norm = nn.LayerNorm(self.d_model)

        self.token_beta_head = nn.Sequential(
            nn.Linear(self.d_model, self.oc_mlp_beta_hidden),
            nn.GELU(),
            nn.Linear(self.oc_mlp_beta_hidden, 1),
            nn.Sigmoid(),
        )
        self.token_pos_head = nn.Sequential(
            nn.Linear(self.d_model, self.oc_mlp_pos_hidden),
            nn.GELU(),
            nn.Dropout(self.oc_mlp_dropout),
            nn.Linear(self.oc_mlp_pos_hidden, self.oc_mlp_pos_hidden),
            nn.GELU(),
            nn.Linear(self.oc_mlp_pos_hidden, self.pos_dim),
        )
        self.refined_score_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid(),
        )
        self.refined_time_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid(),
        )
        self.refined_charge_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
            nn.Softplus(),
        )

        self.last_proposal_score = None
        self.last_proposal_time = None
        self.last_proposal_width = None
        self.last_proposal_amplitude = None
        self.last_proposal_embedding = None
        self.last_token_beta = None
        self.last_token_mask = None
        self.last_token_time = None
        self.last_token_weight = None
        self.last_refined_score = None
        self.last_refined_time = None
        self.last_refined_charge = None
        self.last_cluster_z = None

    def _broadcast_geometry(self, pos: torch.Tensor, batch_size: int) -> torch.Tensor:
        return torch.broadcast_to(pos, (batch_size, pos.shape[-2], pos.shape[-1]))

    def _build_slot_embedding(
        self, batch_size: int, num_nodes: int, device: torch.device
    ) -> torch.Tensor:
        slot_ids = torch.arange(self.num_pulse_tokens, device=device)
        return self.slot_embed(slot_ids).view(1, 1, self.num_pulse_tokens, -1).expand(
            batch_size, num_nodes, -1, -1
        )

    def propose_pulses(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        fea_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_nodes = x.shape[:2]
        pos = self._broadcast_geometry(pos, batch_size)

        token_embed, block_feature, token_time0, token_mask = self.tokenizer(x, fea_mask)
        token_embed = self.token_proj(token_embed)
        block_feature = self.token_proj(block_feature)
        token_mask = token_mask & node_mask.unsqueeze(-1)

        geo_embed = self.geo_mlp(pos).unsqueeze(2)
        slot_embed = self._build_slot_embedding(batch_size, num_nodes, x.device)
        token_embed = token_embed + block_feature.unsqueeze(2) + geo_embed + slot_embed
        token_embed = self.proposal_fuse(token_embed)
        token_embed = token_embed * token_mask.unsqueeze(-1).to(token_embed.dtype)

        pulse_score = self.proposal_score_head(token_embed).squeeze(-1)
        pulse_time = self.proposal_time_head(token_embed).squeeze(-1)
        pulse_width = self.proposal_width_head(token_embed).squeeze(-1)
        pulse_amplitude = self.proposal_amplitude_head(token_embed).squeeze(-1)

        pulse_score = pulse_score * token_mask.to(pulse_score.dtype)
        pulse_time = pulse_time * token_mask.to(pulse_time.dtype)
        pulse_width = pulse_width * token_mask.to(pulse_width.dtype)
        pulse_amplitude = pulse_amplitude * token_mask.to(pulse_amplitude.dtype)

        proposal = {
            "pulse_embedding": token_embed,
            "pulse_score": pulse_score,
            "pulse_time": pulse_time,
            "pulse_width": pulse_width,
            "pulse_amplitude": pulse_amplitude,
            "token_mask": token_mask,
            "pos": pos,
            "base_token_time": token_time0,
        }
        return proposal

    def build_pruning_mask(
        self,
        pulse_score: torch.Tensor,
        token_mask: torch.Tensor,
        score_threshold: float = 0.5,
        top_m: int = 0,
    ) -> torch.Tensor:
        flat_score = pulse_score.reshape(pulse_score.shape[0], -1)
        threshold_mask = pulse_score > score_threshold
        topk_scores, topk_idx = torch.topk(
            flat_score,
            k=min(max(top_m, 1), flat_score.shape[-1]),
            dim=-1,
        )
        topk_mask = torch.zeros_like(flat_score, dtype=torch.bool)
        topk_mask.scatter_(1, topk_idx, topk_scores > -1.0)
        topk_mask = topk_mask.reshape_as(pulse_score)
        top_m_tensor = pulse_score.new_tensor(float(top_m))
        use_topk = (top_m_tensor > 0).to(dtype=torch.bool)
        keep_mask = threshold_mask | (topk_mask & use_topk)
        return keep_mask & token_mask

    def cluster_pulses(
        self,
        proposal: dict[str, torch.Tensor],
        prune_mask: torch.Tensor | None = None,
        soft_pruning: bool = True,
    ) -> dict[str, torch.Tensor]:
        token_embed = proposal["pulse_embedding"]
        pulse_score = proposal["pulse_score"]
        pulse_time = proposal["pulse_time"]
        pulse_amplitude = proposal["pulse_amplitude"]
        token_mask = proposal["token_mask"]
        pos = proposal["pos"]

        score_gate = pulse_score * token_mask.to(pulse_score.dtype)
        hard_gate = token_mask if prune_mask is None else (token_mask & prune_mask)
        mix_gate = score_gate if soft_pruning else hard_gate.to(score_gate.dtype)
        mix_mask = token_mask if soft_pruning else hard_gate

        token_embed = token_embed + self.time_embed(pulse_time.unsqueeze(-1))
        token_embed = token_embed + self.amp_embed(pulse_amplitude.unsqueeze(-1))
        flat_embed = token_embed.reshape(token_embed.shape[0], -1, self.d_model)
        flat_pos = pos.unsqueeze(2).expand(-1, -1, self.num_pulse_tokens, -1).reshape(
            token_embed.shape[0], -1, self.pos_dim
        )
        flat_time = pulse_time.reshape(pulse_time.shape[0], -1)
        flat_mask = mix_mask.reshape(mix_mask.shape[0], -1)
        flat_gate = mix_gate.reshape(mix_gate.shape[0], -1)

        for block in self.cluster_blocks:
            flat_embed = block(flat_embed, flat_pos, flat_time, flat_mask, flat_gate)

        flat_embed = self.cluster_norm(flat_embed)
        clustered = flat_embed.reshape_as(token_embed)

        cluster_beta = self.token_beta_head(clustered).squeeze(-1)
        cluster_z = self.token_pos_head(clustered)
        refined_score = self.refined_score_head(clustered).squeeze(-1)
        refined_time = self.refined_time_head(clustered).squeeze(-1)
        refined_charge = self.refined_charge_head(clustered).squeeze(-1)

        gate = mix_mask.to(cluster_beta.dtype)
        cluster_beta = cluster_beta * gate
        cluster_z = cluster_z * gate.unsqueeze(-1)
        refined_score = refined_score * gate
        refined_time = refined_time * gate
        refined_charge = refined_charge * gate

        return {
            "cluster_seedness_beta": cluster_beta,
            "latent_cluster_coordinate_z": cluster_z,
            "refined_pulse_score": refined_score,
            "refined_time": refined_time,
            "refined_charge": refined_charge,
            "cluster_token_mask": mix_mask,
            "cluster_token_gate": mix_gate,
        }

    def _pool_legacy_outputs(
        self,
        cluster_outputs: dict[str, torch.Tensor],
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        refined_score = cluster_outputs["refined_pulse_score"]
        cluster_beta = cluster_outputs["cluster_seedness_beta"]
        cluster_z = cluster_outputs["latent_cluster_coordinate_z"]
        cluster_mask = cluster_outputs["cluster_token_mask"]

        token_weight = refined_score * cluster_mask.to(refined_score.dtype)
        token_weight = token_weight / token_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        x_c = (cluster_z * token_weight.unsqueeze(-1)).sum(dim=2)
        beta = cluster_beta.max(dim=-1, keepdim=True).values

        x_c = x_c * node_mask.unsqueeze(-1).to(x_c.dtype)
        beta = beta * node_mask.unsqueeze(-1).to(beta.dtype)
        return x_c, beta

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        fea_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proposal = self.propose_pulses(x, pos, fea_mask, node_mask)
        cluster_outputs = self.cluster_pulses(
            proposal=proposal,
            prune_mask=None,
            soft_pruning=True,
        )
        x_c, beta = self._pool_legacy_outputs(cluster_outputs, node_mask)

        self.last_proposal_score = proposal["pulse_score"]
        self.last_proposal_time = proposal["pulse_time"]
        self.last_proposal_width = proposal["pulse_width"]
        self.last_proposal_amplitude = proposal["pulse_amplitude"]
        self.last_proposal_embedding = proposal["pulse_embedding"]
        self.last_token_beta = cluster_outputs["cluster_seedness_beta"]
        self.last_token_mask = proposal["token_mask"]
        self.last_token_time = proposal["pulse_time"]
        self.last_token_weight = cluster_outputs["cluster_token_gate"]
        self.last_refined_score = cluster_outputs["refined_pulse_score"]
        self.last_refined_time = cluster_outputs["refined_time"]
        self.last_refined_charge = cluster_outputs["refined_charge"]
        self.last_cluster_z = cluster_outputs["latent_cluster_coordinate_z"]

        return x_c, beta
