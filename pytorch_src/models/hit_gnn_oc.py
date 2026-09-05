import torch
from torch import nn

from base.model import BaseModel
from layers.oc import ObjectCondensation
from layers.gnn import GravNet



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


class BatchHitGnnObjectCondensationModel(BaseModel):
    """
    Object Condensation model for hit-level data.
    """

    def __init__(self, **kwargs):
        super(BatchHitGnnObjectCondensationModel, self).__init__()

        self.d_model = kwargs.get('d_model', 32)

        # gravnet parameters
        self.n_gravnet_layers = kwargs.get('n_gravnet_layers', 2)
        self.gravnet_space_dimensions = kwargs.get('gravnet_space_dimensions', 4)
        self.gravnet_propagate_dimensions = kwargs.get('gravnet_propagate_dimensions', 22)
        self.gravnet_k = kwargs.get('gravnet_k', 8)

        # oc mlp parameters
        self.oc_mlp_pos_nlayers = kwargs.get('oc_mlp_pos_nlayers', 2)
        self.oc_mlp_pos_hidden = kwargs.get('oc_mlp_pos_hidden', 64)
        self.oc_mlp_pos_out = kwargs.get('oc_mlp_pos_out', 2) 
        self.oc_mlp_dropout = kwargs.get('oc_mlp_dropout', 0.1)
        self.oc_mlp_beta_nlayers = kwargs.get('oc_mlp_beta_nlayers', 2)
        self.oc_mlp_beta_hidden = kwargs.get('oc_mlp_beta_hidden', 64)

        self.out_nlayers = kwargs.get('out_nlayers', 2)
        self.out_hidden = kwargs.get('out_hidden', 64)  # default hidden size for output MLP

        ################################################################################
        # Encoder for input features
        ################################################################################
        self.fea_encoder = nn.Linear(in_features=4, out_features=self.d_model)  # e, t, x, y

        gravnet_layers = [
            GravNet(
                in_channels=self.d_model,
                out_channels=self.d_model,
                space_dimensions=self.gravnet_space_dimensions,
                propagate_dimensions=self.gravnet_propagate_dimensions,
                k=self.gravnet_k,
            )
            for _ in range(self.n_gravnet_layers)
        ]

        self.gnn = nn.ModuleList(gravnet_layers)

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
        self, x: torch.Tensor, pos:torch.Tensor, batch: torch.Tensor
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

        x = torch.cat([x, pos], dim=-1)  # Concatenate input features with positional information
        x = self.fea_encoder(x)  # [N, d_model]

        for gl in range(self.n_gravnet_layers):
            x = self.gnn[gl](x, batch=batch)  # [N, d_model]

        x_signal = self.signal_out(x)  # [N, 1], logits for triggered hits
        x_c, beta = self.oc(x)  # [N, pos_dim], [N, 1]


        return x_c, beta, x_signal
