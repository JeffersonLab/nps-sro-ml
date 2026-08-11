import torch
from torch import nn


class ObjectCondensation(torch.nn.Module):

    """
    Object Condensation module for learning latent space coordinates and beta values for nodes in a graph.
    """

    def __init__(
        self,
        n_x_layers: int = 3,
        x_in: int = 4,
        x_hidden: int = 64,
        x_out: int = 2,
        n_beta_layers: int = 1,
        beta_hidden: int = 32,
        x_dropout: float = 0.1,
    ):
        super(ObjectCondensation, self).__init__()

        self.oc_x = nn.ModuleList()
        self.oc_beta = nn.ModuleList()

        current_dim = x_in
        for _ in range(n_x_layers):
            self.oc_x.append(
                nn.Sequential(
                    nn.Linear(current_dim, x_hidden),
                    nn.ReLU(),
                    nn.Dropout(x_dropout),
                )
            )
            current_dim = x_hidden

        self.oc_x_out = nn.Linear(current_dim, x_out)

        current_beta_dim = x_in
        for _ in range(n_beta_layers):
            self.oc_beta.append(
                nn.Sequential(
                    nn.Linear(current_beta_dim, beta_hidden),
                    nn.ReLU(),
                )
            )
            current_beta_dim = beta_hidden

        self.oc_beta_out = nn.Sequential(
            nn.Linear(current_beta_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the output of the Object Condensation module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, x_in).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - x_c: Tensor of shape (batch_size, seq_len, x_out) representing the latent space coordinates of each node.
            - x_beta: Tensor of shape (batch_size, seq_len, 1) representing the beta values for each node, indicating the likelihood of being a condensation point.

        """
        beta = x

        for layer in self.oc_x:
            x = layer(x)

        x_c = self.oc_x_out(x)

        for layer in self.oc_beta:
            beta = layer(beta)

        x_beta = self.oc_beta_out(beta)

        return x_c, x_beta
