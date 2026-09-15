import torch
from torch import nn

from base.model import BaseModel
from layers.attention import FullAttention, AttentionLayer
from layers.encoders import Encoder, VanillaEncoderLayer
from layers.oc import ObjectCondensation
from layers.gnn import GravNet
from utils.masks import cross_graph_mask


class HitAttnGnnOcModel(BaseModel):
    """
    Object Condensation model for hit-level data.
    """

    def __init__(self, **kwargs):
        super(HitAttnGnnOcModel, self).__init__()

        self.pos_dim = kwargs.get('pos_dim', 2)  # dim of detector position (x,y)

        # gravnet parameters
        self.n_gravnet_layers = kwargs.get('n_gravnet_layers', 2)
        self.gravnet_space_dimensions = kwargs.get('gravnet_space_dimensions', 4)
        self.gravnet_propagate_dimensions = kwargs.get(
            'gravnet_propagate_dimensions', 22
        )
        self.gravnet_k = kwargs.get('gravnet_k', 8)

        # node level encoder parameters
        self.d_model = kwargs.get('d_model', 32)
        self.n_enc_layers = kwargs.get('n_enc_layers', 2)
        self.num_heads = kwargs.get('num_heads', 4)
        self.attn_dropout = kwargs.get('attn_dropout', 0.1)
        self.attn_ff = kwargs.get('attn_ff', self.d_model * 4)

        # oc mlp parameters
        self.oc_mlp_pos_nlayers = kwargs.get('oc_mlp_pos_nlayers', 2)
        self.oc_mlp_pos_hidden = kwargs.get('oc_mlp_pos_hidden', 64)
        self.oc_mlp_dropout = kwargs.get('oc_mlp_dropout', 0.1)
        self.oc_mlp_beta_nlayers = kwargs.get('oc_mlp_beta_nlayers', 2)
        self.oc_mlp_beta_hidden = kwargs.get('oc_mlp_beta_hidden', 64)

        ################################################################################
        # Encoder for input features
        ################################################################################
        self.fea_encoder = nn.Linear(in_features=2, out_features=self.d_model)  # e, t
        self.geo_mlp = nn.Linear(in_features=2, out_features=self.d_model)  # x, y
        ################################################################################
        # Attentional encoder layers
        ################################################################################
        gravnet_layers = [
            GravNet(
                in_channels=self.d_model,
                out_channels=self.d_model,
                space_dimensions=self.gravnet_space_dimensions,
                propagate_dimensions=self.gravnet_propagate_dimensions,
                k=self.gravnet_k,
            )
            for i in range(self.n_gravnet_layers)
        ]

        self.geo_embed = nn.ModuleList(gravnet_layers)

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
            n_x_layers=self.oc_mlp_pos_nlayers,
            x_in=self.d_model,
            x_hidden=self.oc_mlp_pos_hidden,
            x_out=self.pos_dim,
            n_beta_layers=self.oc_mlp_beta_nlayers,
            beta_hidden=self.oc_mlp_beta_hidden,
            x_dropout=self.oc_mlp_dropout,
        )

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (num_nodes, 2) representing (energy, time)
        pos : torch.Tensor
            Geometric positions of shape (num_nodes, 2) representing (x, y)
        batch : torch.Tensor
            Batch vector of shape (num_nodes,) indicating the graph index for each node.
        """

        x = self.fea_encoder(x)  # [N, d_model]

        pos = self.geo_mlp(pos)  # [N, d_model]
        for gl in range(self.n_gravnet_layers):
            pos = self.geo_embed[gl](pos, batch=batch)  # [N, d_model]
        x = x + pos  # [N, d_model]
        x = x.unsqueeze(0)  # [1, N, d_model]

        attn_mask = cross_graph_mask(batch)  # [1, 1, N, N]

        x, _ = self.attn_enc(x, attn_mask=attn_mask)  # [1, N, d_model]
        x = x.squeeze(0)  # [N, d_model]

        x_c, beta = self.oc(x)  # [N, pos_dim], [N, 1]
        return x_c, beta
