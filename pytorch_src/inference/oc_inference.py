from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping

import pandas as pd
import torch

from utils.graph import create_unique_object_ids
from base.dataloader import BaseDataLoader


@dataclass
class BaseOcInferenceHyperparameters:
    """
    Base Object Condensation inference hyperparameters.
    """

    beta_thres: float = 0.5
    dist_thres: float = 0.5
    empty_idx: int = -1

    @classmethod
    def from_mapping(
        cls, params: Mapping[str, Any]
    ) -> "BaseOcInferenceHyperparameters":
        """Construct hyperparameters from recognized mapping entries."""
        mapping = {
            key: value
            for key, value in params.items()
            if key in cls.__dataclass_fields__
        }
        return cls(**mapping)


@dataclass
class BaseOcInferenceResultsPerGraph:
    """
    Container for Object Condensation inference results for 1 event.
    """

    # original data
    event_id: int
    x: torch.Tensor  # [num_nodes, features]
    pos: torch.Tensor  # [num_nodes, detector_dimensions]
    truth_ids: torch.Tensor  # [num_nodes]

    # inference
    min_d: torch.Tensor  # [num_nodes]
    beta: torch.Tensor  # [num_nodes]
    x_c: torch.Tensor  # [num_nodes, latent_dim]
    object_ids: torch.Tensor  # [num_nodes]

    @classmethod
    def from_mapping(
        cls, params: Mapping[str, Any]
    ) -> "BaseOcInferenceResultsPerGraph":
        """Construct a result from recognized mapping entries."""
        mapping = {
            key: value
            for key, value in params.items()
            if key in cls.__dataclass_fields__
        }
        return cls(**mapping)


@dataclass
class BaseOcInferenceResults:
    """
    Container for Object Condensation inference results for multiple events.
    """

    result_type: ClassVar[type[BaseOcInferenceResultsPerGraph]] = (
        BaseOcInferenceResultsPerGraph
    )
    results: list[BaseOcInferenceResultsPerGraph] = field(default_factory=list)

    def append(self, **kwargs) -> None:
        """Append one event using the configured per-graph result type."""
        self.results.append(self.result_type.from_mapping(kwargs))

    @staticmethod
    def _as_rows(value: Any, num_nodes: int, field_name: str) -> torch.Tensor:
        """Convert a result value to one row per node."""
        tensor = (
            value.detach().cpu()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value)
        )

        if tensor.ndim == 0 or tensor.numel() == 1:
            return tensor.reshape(1, 1).expand(num_nodes, 1)
        if tensor.shape[0] != num_nodes:
            raise ValueError(
                f"Result field '{field_name}' has {tensor.shape[0]} rows; "
                f"expected {num_nodes}."
            )
        return tensor.reshape(num_nodes, -1)

    def to_dict(self) -> dict[str, Any]:
        """Return node-level results with dataclass field names as headers."""
        if not self.results:
            return {}

        columns: dict[str, list[torch.Tensor]] = {}
        widths: dict[str, int] = {}

        for result in self.results:
            num_nodes = result.x.shape[0]
            for result_field in fields(result):
                name = result_field.name
                rows = self._as_rows(getattr(result, name), num_nodes, name)
                width = rows.shape[1]
                if name in widths and widths[name] != width:
                    raise ValueError(
                        f"Result field '{name}' has inconsistent flattened widths."
                    )
                widths[name] = width
                columns.setdefault(name, []).append(rows)

        exported: dict[str, Any] = {}
        for name, tensors in columns.items():
            values = torch.cat(tensors, dim=0)
            if widths[name] == 1:
                exported[name] = values[:, 0].numpy()
            else:
                for index in range(widths[name]):
                    exported[f"{name}_{index}"] = values[:, index].numpy()
        return exported

    def to_df(self) -> pd.DataFrame:
        """Return node-level results as a DataFrame."""
        return pd.DataFrame(self.to_dict())

    def to_csv(self, filename: str | Path, **kwargs) -> None:
        """Write node-level results to CSV."""
        kwargs.setdefault("index", False)
        self.to_df().to_csv(filename, **kwargs)


class BaseOcInferenceManager(ABC):
    """Run common Object Condensation inference for a graph dataloader."""

    hyperparameters_type: ClassVar[type[BaseOcInferenceHyperparameters]] = (
        BaseOcInferenceHyperparameters
    )
    results_type: ClassVar[type[BaseOcInferenceResults]] = BaseOcInferenceResults

    def __init__(
        self,
        model: torch.nn.Module,
        hyperparameters: (
            BaseOcInferenceHyperparameters | Mapping[str, Any] | None
        ) = None,
    ):
        """Initialize the inference manager.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose first two outputs are latent positions and condensation
            strengths.
        hyperparameters : BaseOcInferenceHyperparameters or mapping, optional
            Inference settings. Recognized mapping entries are converted to the
            manager's configured hyperparameter type.
        """
        self.model = model
        if hyperparameters is None:
            self.hyperparameters = self.hyperparameters_type()
        elif isinstance(hyperparameters, Mapping):
            self.hyperparameters = self.hyperparameters_type.from_mapping(
                hyperparameters
            )
        else:
            self.hyperparameters = hyperparameters

        self.event_idx = 0
        self.results = self.results_type()

    @abstractmethod
    def _prepare_model_inputs(self, data: Any) -> tuple[torch.Tensor, ...]:
        """
        Prepare the model inputs from the data object. This method should be implemented in subclasses to extract the necessary features from the data object and return them as a tuple of tensors.

        Parameters
        ----------
        data : Any
            The data object containing the input features. Must contain attributes `y`, `batch`.

        """
        raise NotImplementedError

    @abstractmethod
    def _infer(
        self,
        *model_outputs: torch.Tensor,
    ) -> Mapping[str, Any]:
        """
        Perform additional inference for a single graph. This method should be implemented in subclasses to extract any additional information from the model outputs and return it as a mapping of field names to values.

        Parameters
        ----------
        model_outputs : tuple[torch.Tensor, ...]
            The outputs from the model evaluation. The first two elements are expected to be the latent positions and condensation strengths.
        """
        raise NotImplementedError

    def _evaluate_model(
        self, *model_inputs: Any, batch: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Evaluate the model and normalize the standard OC outputs."""
        model_outputs = self.model(*model_inputs, batch=batch)
        if not isinstance(model_outputs, (tuple, list)) or len(model_outputs) < 2:
            raise ValueError("The model must return at least (x_c, beta).")
        x_c, beta = model_outputs[0], model_outputs[1]
        beta = beta.squeeze(-1) if beta.ndim > 1 else beta
        return x_c, beta, *model_outputs[2:]

    def _extract_truth_labels(self, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Create unique truth IDs and the graph-membership vector."""
        y = data.y.squeeze(-1).long()
        batch = (
            data.batch
            if hasattr(data, "batch")
            else torch.zeros(y.shape[0], dtype=torch.long, device=y.device)
        )

        empty_idx = self.hyperparameters.empty_idx
        truth_ids = create_unique_object_ids(y, batch, empty_idx)
        return truth_ids, batch

    def _infer_batch(self, data: Any) -> None:
        """
        Perform inference on a batch of graphs.
        """
        truth_ids, batch = self._extract_truth_labels(data)
        model_inputs = self._prepare_model_inputs(data)
        model_outputs = self._evaluate_model(*model_inputs, batch=batch)

        for b in batch.unique(sorted=True):
            b_mask = batch == b
            b_model_outputs = tuple(output[b_mask] for output in model_outputs)

            inferred_attrs = self._infer(*b_model_outputs)
            self.results.append(
                # input data
                event_id=self.event_idx,
                x=data.x[b_mask],
                pos=data.pos[b_mask],
                truth_ids=truth_ids[b_mask],
                # oc inferences
                **inferred_attrs,
            )
            self.event_idx += 1

    def infer(self, dataloader: BaseDataLoader) -> BaseOcInferenceResults:
        """
        Run the inference of the model on the provided dataloader and return the results.

        Parameters
        ----------
        dataloader : BaseDataLoader
            A dataloader that yields batches of graph data. Requires that each batch has attributes `y` and `batch` for truth labels and graph membership.

        Returns
        -------
        BaseOcInferenceResults
            The results of the inference.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                self._infer_batch(data)
        return self.results


def oc_inference_per_batch(
    x: torch.Tensor,
    beta: torch.Tensor,
    batch: torch.Tensor,
    beta_thres: float = 0.4,
    dist_thres: float = 0.8,
    empty_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform clustering in latent space based on beta and distance thresholding for a batch of graphs.

    Parameters
    ----------
    x : torch.Tensor
        Latent space positions, shape [num_nodes, d_model]
    beta : torch.Tensor
        Condensation strengths, shape [num_nodes] or [num_nodes, 1]
    batch : torch.Tensor
        Batch vector indicating graph membership of each node, shape [num_nodes]
    beta_thres : float, optional
        Minimum beta to be considered for clustering, by default 0.4
    dist_thres : float, optional
        Maximum distance to cluster points together, by default 0.8
    empty_idx : int, optional
        Index to use for background/unassigned, by default 0

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple of tensors in the original node order:
        - cluster_ids: Tensor of shape [num_nodes] with cluster assignments
        - min_d: Tensor of shape [num_nodes] with minimum distance to assigned cluster center
    """
    beta_ = beta.view(-1)

    seed_mask = (beta_ > beta_thres).nonzero(as_tuple=True)[0]  # [S]

    obj_ids = torch.full(
        (x.size(0),),
        fill_value=empty_idx,
        dtype=torch.long,
        device=x.device,
    )
    min_d = torch.full(
        (x.size(0),),
        fill_value=float("inf"),
        dtype=x.dtype,
        device=x.device,
    )

    if len(seed_mask) == 0:
        return obj_ids, min_d

    batch_mask = batch[:, None] == batch[seed_mask][None, :]

    d = torch.cdist(x, x[seed_mask], p=2)  #   [N, S]
    d[~batch_mask] = float("inf")
    has_seed = batch_mask.any(dim=1)

    # assign object ID to all nodes based on closest seed
    min_d[has_seed], obj_ids[has_seed] = d[has_seed].min(dim=1)

    obj_ids[has_seed] += empty_idx + 1  # [N]

    # replulsion based on distance threshold
    obj_ids[min_d > dist_thres] = empty_idx  # background

    return obj_ids, min_d


def oc_inference_per_graph(
    x: torch.Tensor,
    beta: torch.Tensor,
    beta_thres: float = 0.4,
    dist_thres: float = 0.8,
    empty_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform clustering in latent space based on beta and distance thresholding.

    Parameters
    ----------
    x : torch.Tensor
        Latent space positions, shape [num_nodes, d_model]
    beta : torch.Tensor
        Condensation strengths, shape [num_nodes] or [num_nodes, 1]
    beta_thres : float, optional
        Minimum beta to be considered for clustering, by default 0.4
    dist_thres : float, optional
        Maximum distance to cluster points together, by default 0.8
    empty_idx : int, optional
        Index to use for unassigned nodes, by default -1

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        cluster_ids: Tensor of shape [num_nodes] with cluster assignments
        min_d: Tensor of shape [num_nodes] with minimum distance to assigned cluster center
    """
    beta_ = beta.view(-1)
    seed_mask = (beta_ > beta_thres).nonzero(as_tuple=True)[0]  # [S]

    if len(seed_mask) == 0:
        return (
            torch.full(
                (x.size(0),), fill_value=empty_idx, dtype=torch.long, device=x.device
            ),
            torch.full(
                (x.size(0),), fill_value=float("inf"), dtype=x.dtype, device=x.device
            ),
        )

    d = torch.cdist(x, x[seed_mask], p=2)  #   [N, S]

    # assign object ID to all nodes based on closest seed
    min_d, obj_ids = torch.min(d, dim=1)
    # shift to avoid conflicts of empty_idx with valid idx
    obj_ids = obj_ids + empty_idx + 1
    # replulsion based on distance threshold
    obj_ids[min_d > dist_thres] = empty_idx

    return obj_ids, min_d
