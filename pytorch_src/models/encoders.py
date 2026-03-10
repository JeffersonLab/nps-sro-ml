import torch
from torch import nn

from base.model import BaseModel
from layers.attention import FullAttention, AttentionLayer
from layers.encoders import Encoder, VanillaEncoderLayer
from layers.embed import PositionalEmbedding


class WaveformEncoder(BaseModel):
    """Attention-based encoder for waveforms (sequences)."""

    def __init__(self, **kwargs):
        super().__init__()

        d_model = kwargs.get('d_model', 32)
        n_layers = kwargs.get('n_layers', 2)
        n_heads = kwargs.get('n_heads', 4)
        dropout = kwargs.get('dropout', 0.1)
        ff_mult = kwargs.get('ff_mult', 4)
        out_dim = kwargs.get('out_dim', d_model)

        self.embed = nn.Linear(1, d_model)
        self.pe = PositionalEmbedding(d_model=d_model)

        self.layers = nn.ModuleList(
            [
                VanillaEncoderLayer(
                    attn_layer=AttentionLayer(
                        FullAttention(
                            mask_flag=True, attention_dropout=dropout, scale=None
                        ),
                        d_model=d_model,
                        n_heads=n_heads,
                    ),
                    d_model=d_model,
                    ff_kwargs={"d_ff": d_model * ff_mult, "activation": nn.LeakyReLU},
                    dropout=dropout,
                    batchnorm=False,
                )
                for _ in range(n_layers)
            ]
        )
        self.enc = Encoder(self.layers)
        self.proj = nn.Conv1d(d_model, out_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of waveform encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input waveform tensor of shape (G, N, T), where G is batch size, N is max number of nodes per graph, and T is waveform length.
        mask : torch.Tensor
            Tensor of shape (G, N, T) indicating valid sample of the waveform.

        """
        x_out = x.unsqueeze(-1)  # (G, N, T) -> (G, N, T, 1)
        x_out = x_out.reshape(-1, x.shape[-1], 1)  # (G*N, T, 1)

        x_out = self.embed(x_out)  # (G*N, T, 1) -> (G*N, T, d_model)
        x_out = self.pe(x_out)

        mask = mask.reshape(-1, x.shape[-1])  # (G*N, T)
        wf_mask = ~mask
        wf_mask = wf_mask.unsqueeze(1).unsqueeze(1)  # (G*N, 1, 1, T)

        x_out, _ = self.enc(x_out, attn_mask=wf_mask)

        x_out = x_out.permute(0, 2, 1)  # (G*N, T, d_model) -> (G*N, d_model, T)
        x_out = self.proj(x_out)  # (G*N, d_model, T) -> (G*N, out_dim, T)
        # global average pooling over waveform length -> (G*N, out_dim)
        x_out = x_out.mean(dim=-1)

        # (G*N, out_dim) -> (G, N, out_dim)
        return x_out.reshape(x.shape[0], x.shape[1], x_out.shape[-1])


class PulseSetEncoder(BaseModel):
    """
    Encoder for a set of pulses associated with a single node. Each node has a variable number of pulses (up to K_max), each with features (e.g., time, charge). The encoder produces a fixed-size embedding for each node by processing the set of pulses in a permutation-invariant way.
    """

    def __init__(self, hidden: int = 64, out_dim: int = 32):
        super().__init__()

        # pulse-level embedding φ
        self.phi = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # block-level compression ρ
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

        self.ln = nn.LayerNorm(out_dim)

    def forward(self, pulses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute a fixed-size embedding for a set of pulses associated with a single node.

        Parameters
        ----------
        pulses : torch.Tensor
            Tensor of shape [..., K_max, 2] representing the pulses (E, t).
        mask : torch.Tensor
            Tensor of shape [..., K_max] indicating valid pulses (True for valid pulse).
        """
        x = self.phi(pulses)  # [..., K_max, hidden]
        x = x * mask.unsqueeze(-1)  # zero out padded pulses

        x = x.sum(dim=-2)  # permutation invariant aggregation
        x = self.rho(x)  # [..., out_dim]
        x = self.ln(x)
        return x
