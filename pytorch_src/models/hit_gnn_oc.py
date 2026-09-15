import torch
from torch import nn

from base.model import BaseModel
from layers.oc import ObjectCondensation
from layers.gnn import GravNetConv


class SignalDenseLayer(BaseModel):
    def __init__(self, nlayers: int = 2, input: int = 32, hidden: int = 32):
        super(SignalDenseLayer, self).__init__()

        self.network = nn.ModuleList()

        current_dim = input
        for _ in range(nlayers):
            self.network.append(
                nn.Sequential(
                    nn.Linear(current_dim, hidden),
                    nn.ReLU(),
                )
            )
            current_dim = hidden

        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.network:
            x = layer(x)
        x = self.out(x)
        return x


class HitGnnOcModel(BaseModel):
    """
    Object Condensation model for hit-level data.
    """

    def __init__(self, **kwargs):
        super(HitGnnOcModel, self).__init__()

        self.d_model = kwargs.get('d_model', 32)

        # gravnet parameters
        self.n_gravnet_layers = kwargs.get('n_gravnet_layers', 2)
        self.gravnet_space_dimensions = kwargs.get('gravnet_space_dimensions', 4)
        self.gravnet_propagate_dimensions = kwargs.get(
            'gravnet_propagate_dimensions', 22
        )
        self.gravnet_k = kwargs.get('gravnet_k', 8)
        self.gravnet_scale = kwargs.get('gravnet_scale', 10)

        # oc mlp parameters
        self.oc_mlp_pos_nlayers = kwargs.get('oc_mlp_pos_nlayers', 2)
        self.oc_mlp_pos_hidden = kwargs.get('oc_mlp_pos_hidden', 64)
        self.oc_mlp_pos_out = kwargs.get('oc_mlp_pos_out', 2)
        self.oc_mlp_dropout = kwargs.get('oc_mlp_dropout', 0.1)
        self.oc_mlp_beta_nlayers = kwargs.get('oc_mlp_beta_nlayers', 2)
        self.oc_mlp_beta_hidden = kwargs.get('oc_mlp_beta_hidden', 64)

        self.out_nlayers = kwargs.get('out_nlayers', 2)
        self.out_hidden = kwargs.get(
            'out_hidden', 64
        )  # default hidden size for output MLP

        ################################################################################
        # Encoder for input features
        ################################################################################

        # e and log(e)
        self.energy_encoder = nn.Sequential(
            nn.Linear(2, self.d_model // 4),
            nn.ReLU(),
        )

        self.time_encoder = nn.Sequential(
            nn.Linear(1, self.d_model // 4),
            nn.ReLU(),
        )

        self.geo_encoder = nn.Sequential(
            nn.Linear(2, self.d_model // 2),
            nn.ReLU(),
        )

        self.gnn = GravNetConv(
            in_channels=self.d_model,
            out_channels=self.d_model,
            space_dimensions=self.gravnet_space_dimensions,
            propagate_dimensions=self.gravnet_propagate_dimensions,
            k=self.gravnet_k,
            n_layers=self.n_gravnet_layers,
            scale=self.gravnet_scale,
        )

        self.gnn_fusion = nn.Sequential(
            nn.Linear(self.d_model * self.n_gravnet_layers, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
        )

        self.oc = ObjectCondensation(
            n_x_layers=self.oc_mlp_pos_nlayers,
            x_in=self.d_model,
            x_hidden=self.oc_mlp_pos_hidden,
            x_out=self.oc_mlp_pos_out,
            n_beta_layers=self.oc_mlp_beta_nlayers,
            beta_hidden=self.oc_mlp_beta_hidden,
            x_dropout=self.oc_mlp_dropout,
        )

        self.signal_out = SignalDenseLayer(
            nlayers=self.out_nlayers,
            input=self.d_model,
            hidden=self.out_hidden,
        )

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (num_nodes, 3) representing (scaled e, log(e), time)
        pos : torch.Tensor
            Geometric positions of shape (num_nodes, 2) representing (x, y)
        batch : torch.Tensor
            Batch vector of shape (num_nodes,) indicating the graph index for each node.
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            x_c: [N, pos_dim]
            beta: [N, 1]
            x_signal: [N, 1]
        """

        feat_e = x[:, :2]  # [N, 2] energy features (scaled e, log(e))
        feat_t = x[:, 2:3]  # [N, 1] time feature

        x_e = self.energy_encoder(feat_e)  # [N, d_model//4] energy
        x_t = self.time_encoder(feat_t)  # [N, d_model//4] time
        x_pos = self.geo_encoder(pos)  # [N, d_model//2] position

        x = torch.cat([x_e, x_t, x_pos], dim=-1)  # [N, d_model]

        x = self.gnn(x, batch=batch)  # [N, d_model * n_gravnet_layers]
        x = self.gnn_fusion(x)  # [N, d_model]

        x_signal = self.signal_out(x)  # [N, 1], logits for triggered hits
        x_c, beta = self.oc(x)  # [N, pos_dim], [N, 1]

        return x_c, beta, x_signal
