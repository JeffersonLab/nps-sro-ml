import torch
from torch import nn

from layers.attention import FullAttention, AttentionLayer
from layers.encoders import Encoder, VanillaEncoderLayer
from models.oc_base import ObjectCondensationBaseModel
from models.encoders import PulseSetEncoder, WaveformEncoder


class ObjectCondensationModel(ObjectCondensationBaseModel):
    """
    Base class for Object Condensation models.
    """

    def __init__(self, **kwargs):
        super(ObjectCondensationModel, self).__init__(**kwargs)

        # Model hyperparameters
        self.pos_dim = kwargs.get('pos_dim', 2)  # dim of detector position (x,y)

        # if waveform encoder is used.
        self.wf_enc_d_model = kwargs.get('wf_d_model', 32)
        self.wf_enc_layers = kwargs.get('wf_enc_layers', 2)
        self.wf_enc_heads = kwargs.get('wf_enc_heads', 4)
        self.wf_enc_dropout = kwargs.get('wf_enc_dropout', 0.1)
        # WaveformEncoder expects a feedforward width multiplier, not an absolute size.
        self.wf_enc_ff = kwargs.get('wf_enc_ff', 4)

        # if pulse set encoder is used.
        self.embed_in = kwargs.get('embed_in', 2)
        self.embed_out = kwargs.get('embed_out', 32)

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
        # input encoder
        if self.input_type == "pulse_set":
            self.fea_encoder = PulseSetEncoder(
                hidden=self.embed_in, out_dim=self.embed_out
            )
            # Keep node feature width consistent for geometric fusion and attention.
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

        self.oc_mlp_pos = nn.Sequential(
            nn.Linear(self.d_model, self.oc_mlp_pos_hidden),
            nn.ReLU(),
            nn.Dropout(self.oc_mlp_dropout),
            nn.Linear(self.oc_mlp_pos_hidden, self.oc_mlp_pos_hidden),
            nn.ReLU(),
            nn.Linear(self.oc_mlp_pos_hidden, self.pos_dim),
        )

        self.oc_mlp_beta = nn.Sequential(
            nn.Linear(self.d_model, self.oc_mlp_beta_hidden),
            nn.ReLU(),
            nn.Linear(self.oc_mlp_beta_hidden, 1),  # output: beta
            nn.Sigmoid(),  # ensure beta is in [0, 1]
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
