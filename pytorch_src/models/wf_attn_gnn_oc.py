import torch
from torch import nn

from base.model import BaseModel
from layers.attention import FullAttention, AttentionLayer
from layers.encoders import Encoder, VanillaEncoderLayer
from layers.embed import PositionalEmbedding
from layers.oc import ObjectCondensation


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


class WfAttnGnnOcModel(BaseModel):
    """
    Base class for Object Condensation models.
    """

    def __init__(self, **kwargs):
        super(WfAttnGnnOcModel, self).__init__()

        self.pos_dim = kwargs.get('pos_dim', 2)  # dim of detector position (x,y)

        self.wf_enc_d_model = kwargs.get('wf_d_model', 32)
        self.wf_enc_layers = kwargs.get('wf_enc_layers', 2)
        self.wf_enc_heads = kwargs.get('wf_enc_heads', 4)
        self.wf_enc_dropout = kwargs.get('wf_enc_dropout', 0.1)
        self.wf_enc_ff = kwargs.get('wf_enc_ff', 4)

        # node level encoder parameters
        self.d_model = kwargs.get('d_model', 32)
        self.n_enc_layers = kwargs.get('n_enc_layers', 2)
        self.num_heads = kwargs.get('num_heads', 4)
        self.attn_dropout = kwargs.get('attn_dropout', 0.1)
        self.attn_ff = kwargs.get('attn_ff', self.d_model * 4)

        # oc mlp parameters
        self.oc_mlp_pos_hidden = kwargs.get('oc_mlp_pos_hidden', 64)
        self.oc_mlp_dropout = kwargs.get('oc_mlp_dropout', 0.1)
        self.oc_mlp_beta_hidden = kwargs.get('oc_mlp_beta_hidden', 64)

        ################################################################################
        # Encoder for input features (waveforms or pulse sets)
        ################################################################################
        self.fea_encoder = WaveformEncoder(
            d_model=self.wf_enc_d_model,
            n_layers=self.wf_enc_layers,
            n_heads=self.wf_enc_heads,
            dropout=self.wf_enc_dropout,
            ff_mult=self.wf_enc_ff,
            out_dim=self.d_model,
        )
        self.fea_proj = nn.Identity()

        ################################################################################
        # Embed geometric information
        ################################################################################
        self.geo_mlp = nn.Sequential(
            nn.Linear(self.pos_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        ################################################################################
        # Attentional encoder layers for graph processing
        ################################################################################

        attn_layers = [
            VanillaEncoderLayer(
                attn_layer=AttentionLayer(
                    FullAttention(
                        mask_flag=True,
                        attention_dropout=self.attn_dropout,
                        scale=None,
                    ),
                    d_model=self.d_model,
                    n_heads=self.num_heads,
                ),
                d_model=self.d_model,
                ff_kwargs={
                    "d_ff": self.attn_ff,
                    "activation": nn.LeakyReLU,
                },
                dropout=self.attn_dropout,
                batchnorm=False,
            )
            for _ in range(self.n_enc_layers)
        ]
        self.attn_enc = Encoder(attn_layers)

        self.oc = ObjectCondensation(
            n_x_layers=2,
            x_in=self.d_model,
            x_hidden=self.oc_mlp_pos_hidden,
            x_out=self.pos_dim,
            n_beta_layers=2,
            beta_hidden=self.oc_mlp_beta_hidden,
            x_dropout=self.oc_mlp_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        fea_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward pass for the Object Condensation model.

        Parameters
        ----------
        x : torch.Tensor
            Input features with shape determined by `input_type`:
            - `pulse_set`: (batch_size, num_nodes, num_pulses, 2)
            - `waveform`: (batch_size, num_nodes, waveform_length)
        pos : torch.Tensor
            Geometric positions of shape (batch_size, num_nodes, pos_dim).
        fea_mask : torch.Tensor
            Mask for input features with shape determined by `input_type`:
            - `pulse_set`: (batch_size, num_nodes, num_pulses)
            - `waveform`: (batch_size, num_nodes, waveform_length)
        node_mask : torch.Tensor
            Mask for valid nodes, shape (batch_size, num_nodes).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - pos_out: Predicted positions of shape (batch_size, num_nodes, pos_dim).
            - beta: Condensation scores of shape (batch_size, num_nodes, 1).
        """

        x = self.fea_encoder(x, fea_mask)
        x = self.fea_proj(x)

        pos = self.geo_mlp(pos)  # [G, N, 2] -> [G, N, d_model]

        # add geo positional embedding
        x = x + pos  # [G, N, d_model]

        """
        if scores of (padded Q x padded K) are masked to -inf, softmax will yield NaN
        instead, we create attn mask based on key only -> scores in (padded Q x valid K)
        it is the standard approach as padded nodes should be discarded downstream
        see my gist
        """

        attn_mask = ~node_mask
        attn_mask = attn_mask.unsqueeze(1)  # [B, 1, N_max]
        attn_mask = attn_mask.unsqueeze(2)  # [B, 1, 1, N_max]

        x, _ = self.attn_enc(x, attn_mask=attn_mask)  # [G, N, d_model]

        x_c = self.oc_mlp_pos(x)
        beta = self.oc_mlp_beta(x)

        return x_c, beta
