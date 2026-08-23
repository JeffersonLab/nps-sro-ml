import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Literal, Sequence, Union
from utils.graph import indices_to_edge_index


class kNN(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def _knn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the k-nearest neighbors of each point in x with respect to points in y.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, D], where N is the number of samples and D is the feature dimension.
        y : torch.Tensor
            Reference tensor of shape [M, D], where M is the number of reference samples and D is the feature dimension.

        Returns
        -------
        indices : torch.Tensor
            Tensor of shape [N, k] containing the indices of the k nearest neighbors from y for each sample in x.

        """

        # [N, 1, D] - [1, M, D] -> [N, M, D]
        diff = x[:, None, :] - y[None, :, :]
        dist2 = torch.sum(diff * diff, dim=-1)
        _, indices = torch.topk(
            dist2,
            k=self.k,
            dim=1,
            largest=False,
            sorted=True,
        )
        return indices

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_x: Optional[torch.Tensor] = None,
        batch_y: Optional[torch.Tensor] = None,
        return_edge_index: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, D], where N is the number of samples and D is the feature dimension.
        y : torch.Tensor
            Reference tensor of shape [M, D], where M is the number of reference samples and D is the feature dimension.

        Returns
        -------
        torch.Tensor
            Tensor of shape [N, k, D] containing the k nearest neighbors from y for each sample in x.
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

        # Compute k-nearest neighbors for each batch
        indices = torch.empty((x.size(0), self.k), dtype=torch.long, device=x.device)
        for b in torch.unique(batch_x):
            mask_x = batch_x == b
            mask_y = batch_y == b

            if mask_x.sum() == 0 or mask_y.sum() == 0:
                continue

            x_b = x[mask_x]
            y_b = y[mask_y]
            global_y_indices = mask_y.nonzero(as_tuple=True)[0]

            # minmax scale to [0, 1]
            min_xy = min(x_b.min().item(), y_b.min().item())
            x_b, y_b = x_b - min_xy, y_b - min_xy

            max_xy = max(x_b.max().item(), y_b.max().item())
            eps = 1e-12
            x_b, y_b = x_b / (max_xy + eps), y_b / (max_xy + eps)

            local_indices = self._knn(x_b, y_b)
            indices[mask_x] = global_y_indices[local_indices]

        if return_edge_index:
            return indices_to_edge_index(indices)

        return indices


class MessagePassing(nn.Module, ABC):
    def __init__(
        self,
        aggr: Union[
            Literal['add', 'mean', 'max'],
            Sequence[Literal['add', 'mean', 'max']],
        ] = 'add',
    ):
        super().__init__()
        self.aggr = [aggr] if isinstance(aggr, str) else list(aggr)
        if not self.aggr or any(a not in {'add', 'mean', 'max'} for a in self.aggr):
            raise ValueError(f"Aggregation method(s) ls{self.aggr!r} not supported.")

    def _aggregate(self, messages: torch.Tensor) -> torch.Tensor:
        outputs = []
        for aggr in self.aggr:
            if aggr == 'add':
                outputs.append(messages.sum(dim=0))
            elif aggr == 'mean':
                outputs.append(messages.mean(dim=0))
            else:  # max
                outputs.append(messages.max(dim=0).values)
        return torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def message(self, x_j: torch.Tensor, edge_features, **kwargs) -> torch.Tensor:
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
        size: Optional[tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Orchestrate the message passing.

        Parameters
        ----------
        x : torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Source features, or a ``(source, target)`` pair for bipartite
            message passing.
        size : tuple[int, int], optional
            Number of source and target nodes. Required when target features
            are not provided.
        edge_index : torch.Tensor
            Edge indices of shape [2, E], where E is the number of edges.

        Returns
        -------
        torch.Tensor
            Tensor of shape [N, D] after message passing.
        """
        source, target = (x, x) if isinstance(x, torch.Tensor) else x
        if size is None:
            if target is None:
                raise ValueError("size is required when target features are None.")
            size = (source.size(0), target.size(0))

        row, col = edge_index
        outputs = []
        for i in range(size[1]):

            edge_mask = col == i
            edge_ids = edge_mask.nonzero(as_tuple=True)[0]

            if edge_ids.numel() == 0:
                aggregated = source.new_zeros(source.size(-1) * len(self.aggr))
            else:
                neighbors = row[edge_ids]
                weights = edge_weight[edge_ids]

                # Compute and aggregate messages from source neighbors.
                messages = self.message(source[neighbors], weights, **kwargs)
                aggregated = self._aggregate(messages)

            # Update node embedding
            outputs.append(self.update(aggregated, **kwargs))

        if not outputs:
            raise ValueError("Message passing requires at least one target node.")
        return torch.stack(outputs, dim=0)

    def forward(self, x):
        raise NotImplementedError("...")


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
        super().__init__(aggr=['mean', 'max'])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.space_dimensions = space_dimensions
        self.propagate_dimensions = propagate_dimensions
        self.k = k
        self.scale = scale

        self.knn = kNN(k)
        self.lin_s = nn.Linear(in_channels, space_dimensions)
        self.lin_h = nn.Linear(in_channels, propagate_dimensions)
        self.lin_out1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_out2 = nn.Linear(2 * propagate_dimensions, out_channels)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return x_j * edge_weight.unsqueeze(1)

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def forward(self, x, batch=None):

        is_bipartite: bool = True
        if isinstance(x, torch.Tensor):
            x = (x, x)
            is_bipartite = False

        if batch is None:
            batch = (
                x[0].new_zeros(x[0].size(0), dtype=torch.long),
                x[1].new_zeros(x[1].size(0), dtype=torch.long),
            )

        elif isinstance(batch, torch.Tensor):
            batch = (batch, batch)
        else:
            batch = (batch[0], batch[1])

        h_l = self.lin_h(x[0])

        s_l = self.lin_s(x[0])
        s_r = self.lin_s(x[1]) if is_bipartite else s_l

        edge_index = self.knn(s_r, s_l, batch[1], batch[0], return_edge_index=True)

        edge_weight = (s_l[edge_index[0]] - s_r[edge_index[1]]).pow(2).sum(-1)
        edge_weight = torch.exp(-self.scale * edge_weight)

        out = self.propagate(
            (h_l, None),
            edge_index,
            edge_weight,
            size=(s_l.size(0), s_r.size(0)),
        )

        return self.lin_out1(x[1]) + self.lin_out2(out)
