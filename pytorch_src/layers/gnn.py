from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn as nn
from utils.graph import indices_to_edge_index


def knn(
    k: int,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    return_edge_index: bool = False,
    exclude_self: bool = True,
) -> torch.Tensor:
    """
    Parameters
    ----------
    k : int
        Number of nearest neighbors to find.
    x : torch.Tensor
        Input tensor of shape [N, D], where N is the number of samples and D is the feature dimension.
    y : torch.Tensor
        Reference tensor of shape [M, D], where M is the number of reference samples and D is the feature dimension.
    batch_x : Optional[torch.Tensor], optional
        Batch indices for x, of shape [N]. If None, all samples are assumed to belong to the same batch. Default is None.
    batch_y : Optional[torch.Tensor], optional
        Batch indices for y, of shape [M]. If None, all samples are assumed to belong to the same batch. Default is None.
    return_edge_index : bool, optional
        If True, returns the edge index of the k nearest neighbors instead of the distances and indices. Default is False.

    Returns
    -------
    tuple[torch.Tensor, torch.BoolTensor]
        A tuple containing:
        - Tensor of shape [N, k] containing the indices of the k nearest neighbors from y for each sample in x.
        - Boolean tensor of shape [N, k] indicating valid neighbors.
    """

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    if x.size(1) != y.size(1):
        raise ValueError(
            f"Feature dimensions of x and y must match. Got {x.size(1)} and {y.size(1)}."
        )

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    diff = x[:, None, :] - y[None, :, :]
    dist2 = torch.sum(diff * diff, dim=-1)

    same_batch = batch_x[:, None] == batch_y[None, :]
    dist2 = dist2.masked_fill(~same_batch, float('inf'))

    if exclude_self and x.size(0) == y.size(0):
        self_mask = torch.eye(
            x.size(0),
            dtype=torch.bool,
            device=x.device,
        )
        dist2 = dist2.masked_fill(self_mask, float("inf"))

    distances, indices = torch.topk(
        dist2,
        k=k,
        dim=1,
        largest=False,
        sorted=True,
    )

    valid = torch.isfinite(distances)

    if return_edge_index:
        return indices_to_edge_index(indices, valid)

    return indices, valid


class MessagePassing(nn.Module, ABC):
    def __init__(self, aggr: Sequence[Literal["add", "mean", "max"]]):
        super().__init__()
        self.aggr = aggr

    def _aggregate(self, messages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages from neighbors.

        Parameters
        ----------
        messages : torch.Tensor
            Messages from neighbors, shape [num_edges, message_dim].
        mask : torch.BoolTensor
            Mask indicating which messages belong to which nodes, shape [num_nodes, num_edges].

        **kwargs
            Additional keyword arguments for aggregation.
        Returns
        -------
        torch.Tensor
            Aggregated messages, shape [num_nodes, message_dim].
        """

        outputs = []
        for aggr in self.aggr:

            if aggr == "add":
                ag = torch.matmul(mask.float(), messages)
            elif aggr == "mean":
                n = mask.float().sum(dim=1, keepdim=True)
                sum_ = torch.matmul(mask.float(), messages)
                ag = sum_ / n.clamp(min=1)
            elif aggr == "max":
                expanded_messages = messages.unsqueeze(0)  # [1, E, P]
                expanded_mask = mask.unsqueeze(-1)  # [N, E, 1]
                masked_messages = torch.where(
                    expanded_mask, expanded_messages, torch.tensor(float('-inf'))
                )
                ag = masked_messages.max(dim=1).values
                has_neighbors = mask.any(dim=1, keepdim=True)

                ag = torch.where(
                    has_neighbors,
                    ag,
                    torch.zeros_like(ag),
                )

            outputs.append(ag)

        return torch.cat(outputs, dim=-1)  # [N, P * len(aggr)]

    @abstractmethod
    def message(self, x_j: torch.Tensor, edge_weights, **kwargs) -> torch.Tensor:
        """
        Compute the node output, as represented as `\phi(x_i, x_j, e_{ji})`
        """

    @abstractmethod
    def update(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Update node features in layer k using the results from layer (k-1)

        Parameters
        ----------
        inputs : torch.Tensor
            output in layer (k-1)
        """

    def propagate(
        self,
        x: Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]],
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        source, target = (x, x) if isinstance(x, torch.Tensor) else x

        row, col = edge_index

        messages = self.message(source[row], edge_weight, **kwargs)  # [E, P]
        nodes = torch.arange(
            target.size(0),
            device=messages.device,
        )
        mask = nodes[:, None] == col[None, :]  # [N, E]

        aggregated = self._aggregate(messages, mask)
        return self.update(aggregated, **kwargs)

    def forward(self, x):
        raise NotImplementedError(
            "The forward method must be implemented in subclasses of MessagePassing."
        )


class GravNet(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        space_dimensions: int,
        propagate_dimensions: int,
        k: int,
        scale: float = 10,
    ):
        super().__init__(aggr=["add", "mean", "max"])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.space_dimensions = space_dimensions
        self.propagate_dimensions = propagate_dimensions
        self.k = k
        self.scale = scale

        self.lin_s = nn.Linear(in_channels, space_dimensions)
        self.lin_h = nn.Linear(in_channels, propagate_dimensions)
        self.lin_out1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_out2 = nn.Linear(3 * propagate_dimensions, out_channels)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return x_j * edge_weight.unsqueeze(1)

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def forward(self, x, batch=None):

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h_l = self.lin_h(x)
        s_l = self.lin_s(x)

        edge_index = knn(self.k, s_l, s_l, batch, batch, return_edge_index=True)

        edge_weight = (s_l[edge_index[0]] - s_l[edge_index[1]]).pow(2).sum(-1)
        edge_weight = torch.exp(-self.scale * edge_weight)

        out = self.propagate(
            h_l,
            edge_index,
            edge_weight,
        )

        return self.lin_out1(x) + self.lin_out2(out)
