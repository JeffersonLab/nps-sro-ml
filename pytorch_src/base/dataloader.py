import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Any, Optional, Union

try:
    from torch_geometric.data import Batch as PygBatch
    from torch_geometric.data import Dataset as PygDataset

    Dataset = PygDataset
    HAS_PYG = True
except ImportError:
    PygBatch = None
    HAS_PYG = False


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


class TorchGraphBatch:
    """Minimal batched graph container for the torch-only dataloader path."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to(self, device: torch.device):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self


def _sample_keys(sample: Any) -> list[str]:
    if hasattr(sample, "keys") and callable(sample.keys):
        return [key for key in sample.keys() if getattr(sample, key, None) is not None]
    if hasattr(sample, "__dict__"):
        return [
            key
            for key, value in vars(sample).items()
            if not key.startswith("_") and value is not None
        ]
    return []


def _torch_graph_collate(batch: list[Any]) -> Any:
    if len(batch) == 0:
        return batch

    first = batch[0]
    first_keys = _sample_keys(first)
    if "x" not in first_keys:
        return default_collate(batch)

    keys = sorted(
        {
            key
            for sample in batch
            for key in _sample_keys(sample)
        }
    )

    node_counts = [_as_tensor(sample.x).shape[0] for sample in batch]
    batch_idx = [
        torch.full((count,), idx, dtype=torch.long)
        for idx, count in enumerate(node_counts)
    ]

    collated = {"batch": torch.cat(batch_idx, dim=0)}
    edge_offset = 0

    for key in keys:
        values = [getattr(sample, key, None) for sample in batch]
        values = [value for value in values if value is not None]
        if len(values) == 0:
            continue

        tensors = [_as_tensor(value) for value in values]
        if key == "edge_index":
            shifted = []
            node_offset = 0
            for tensor, count in zip(tensors, node_counts):
                shifted.append(tensor + node_offset)
                node_offset += count
            collated[key] = torch.cat(shifted, dim=1)
            edge_offset += 1
        elif tensors[0].ndim == 0:
            collated[key] = torch.stack(tensors, dim=0)
        else:
            collated[key] = torch.cat(tensors, dim=0)

    return TorchGraphBatch(**collated)


def _pyg_graph_collate(batch: list[Any]) -> Any:
    if not HAS_PYG:
        return _torch_graph_collate(batch)
    return PygBatch.from_data_list(batch)


class BaseDataLoader(TorchDataLoader):
    """
    Base DataLoader with train/validation splitting.

    By default it uses PyG-style collation when PyG is available. Setting
    `use_torch_loader=True` opts into the torch-only collation path even if PyG
    is installed.
    Examples
    --------
    >>> from torch_geometric.datasets import TUDataset
    >>> dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    >>> dataloader = BaseDataLoader(dataset, batch_size=32, validation_split=0.2, shuffle=True)
    >>> val_loader = dataloader.split_validation()
    >>> for batch in dataloader:
    >>>    # Training loop here
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        validation_split: Union[float, int] = 0.0,
        num_workers: int = 0,
        random_seed: int = 0,
        use_torch_loader: bool = False,
    ):
        """
        Initialize the BaseDataLoader.
        Parameters
        ----------
        dataset : Dataset
            The dataset to load data from.
        batch_size : int
            Number of samples per batch.
        shuffle : bool, optional
            Whether to shuffle the dataset at every epoch (default is True). Note that if a validation split is used, shuffling will be disabled.
        validation_split : float or int, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to use for validation.
            If int, represents the absolute number of samples to use for validation.
        num_workers : int, optional
            Number of subprocesses to use for data loading (default is 0).
        random_seed : int, optional
            Random seed for reproducibility (default is 0).

        """

        self.validation_split = validation_split
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.use_torch_loader = use_torch_loader
        self.dataset = dataset

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        # Store init kwargs for creating validation loader
        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": (
                _torch_graph_collate
                if self.use_torch_loader
                else _pyg_graph_collate
            ),
        }

        # When sampler is used, shuffle must be False
        super().__init__(
            sampler=self.sampler,
            shuffle=self.shuffle if self.sampler is None else False,
            **self.init_kwargs,
        )

    def _split_sampler(self, split: Union[float, int]) -> tuple:
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        # Use specified random seed for reproducibility
        np.random.seed(self.random_seed)
        np.random.shuffle(idx_full)

        # Determine validation set size
        if isinstance(split, int):
            assert split > 0, "validation_split must be positive"
            assert (
                split < self.n_samples
            ), f"validation set size ({split}) is larger than dataset ({self.n_samples})"
            len_valid = split
        else:
            assert (
                0.0 < split < 1.0
            ), "validation_split as float must be between 0 and 1"
            len_valid = int(self.n_samples * split)

        # Split indices
        valid_idx = idx_full[:len_valid]
        train_idx = idx_full[len_valid:]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self, **kwargs) -> Optional[TorchDataLoader]:
        """
        Create a validation DataLoader using the validation split.
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the underlying DataLoader constructor. These will override the default initialization arguments.
        Returns
        -------
        Optional[TorchDataLoader]
            A DataLoader for the validation set, or None if no validation split was defined.
        """

        if self.valid_sampler is None:
            return None

        # Create kwargs for validation loader
        val_kwargs = self.init_kwargs.copy()
        val_kwargs.update(kwargs)
        val_kwargs['shuffle'] = False

        # Validation loader uses the valid_sampler and never shuffles
        return TorchDataLoader(sampler=self.valid_sampler, **val_kwargs)

    def get_train_size(self) -> int:
        """Return the number of training samples."""
        return self.n_samples

    def get_val_size(self) -> int:
        """Return the number of validation samples."""
        if self.valid_sampler is None:
            return 0
        return len(self.valid_sampler.indices)
