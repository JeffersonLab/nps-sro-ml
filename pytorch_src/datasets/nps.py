import pathlib
import torch
import logging
from typing import Optional
from torch.utils.data import Dataset as TorchDataset
from base.dataloader import BaseDataLoader
from utils.utils import get_logger
from utils.graph import reindex_edge_index

try:
    from torch_geometric.data import Dataset, Data
except ImportError:
    class Dataset(TorchDataset):
        """Minimal fallback Dataset compatible with the project's usage."""

        def __init__(self, *args, **kwargs):
            super().__init__()

        def __len__(self):
            return self.len()

        def __getitem__(self, idx):
            return self.get(idx)

    class Data:
        """Minimal fallback data container matching the attributes used downstream."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to(self, device):
            for key, value in self.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(self, key, value.to(device))
            return self


NTIME = 110
NCOLS = 30
NROWS = 36


def get_node_index_from_position(row_idx: int, col_idx: int) -> int:
    """Return node index / channel ID from row and column indices."""
    return row_idx * NCOLS + col_idx


def get_position_from_node_index(node_idx: int) -> tuple[int, int]:
    """Return row and column indices from node index / channel ID."""
    row_idx = node_idx // NCOLS
    col_idx = node_idx % NCOLS
    return row_idx, col_idx


class NPSDataset(Dataset):
    """
    Dataset class for NPS SRO data stored in PyTorch Geometric format. The class is designed to load .pt files containing pre-processed graph data with node features representing waveforms from detector channels, see `unpack` method for details.
    """

    def __init__(
        self,
        paths: Optional[list[str | pathlib.Path]] = None,
        data_dir: Optional[pathlib.Path | str] = None,
        logger: Optional[logging.Logger] = None,
        max_files: Optional[int] = None,
        use_edges: bool = False,
    ):
        """
        Initialize NPSDataset, forces `transform`, `pre_transform`, and `pre_filter` to be None. If you want to apply any transformations or filtering, do it before initializing the dataset. Implementing those functionalities in this class lead to confusion. For instance, the `processed_file_names` is expected to be known beforehand, which is not possible if `pre_filter` skips some samples.
        """
        self.logger = get_logger("NPSDataset") if logger is None else logger

        if paths is None and data_dir is None:
            raise ValueError("Either 'paths' or 'data_dir' must be provided.")

        self.paths: list[pathlib.Path] = []
        max_files = max_files if max_files is not None else len(paths)
        if paths is None:
            data_dir = pathlib.Path(data_dir)
            self.logger.info(f"Scanning for .pt files under {data_dir}")

            iterator = data_dir.glob("*.pt")
            for pth in iterator:
                if len(self.paths) >= max_files:
                    break
                if self._validate_file(pth):
                    self.paths.append(pth)
                else:
                    self.logger.warning(f"Invalid data file skipped: {pth}")

            self.logger.info(f"Loading data files from directory: {data_dir}")

        else:
            input_paths = [pathlib.Path(p) for p in paths][:max_files]
            for pth in input_paths:
                if self._validate_file(pth):
                    self.paths.append(pth)
                else:
                    self.logger.warning(f"Invalid data file skipped: {pth}")

        if len(self.paths) == 0:
            raise RuntimeError("No valid .pt files found.")

        root = self.paths[0].parent / ".pyg"
        self.use_edges = use_edges
        super().__init__(root=root, transform=None, pre_transform=None, pre_filter=None)

    def _validate_file(self, path: pathlib.Path) -> bool:
        """Validate if the given file is a valid .pt data file."""
        return (
            path.suffix == ".pt"
            and path.is_file()
            and path.exists()
            and path.stat().st_size > 0
        )

    @property
    def raw_file_names(self):
        """Returns an empty list. Included to satisfy PyG Dataset requirements."""
        return []

    @property
    def processed_file_names(self):
        """Returns an empty list. Included to satisfy PyG Dataset requirements."""
        return []

    def download(self):
        """Skip downloading as data are assumed to be locally available."""
        self.logger.info(
            f"Skip downloading as data are locally available at {self.paths[0].parent}."
        )

    def process(self):
        """Skip processing as data are assumed to be pre-processed."""
        self.logger.info("Skip processing as data are pre-processed.")

    def _unpack_data(
        self, data: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpack raw data into components.

        The expected structure of the input data tuple is as follows:
        0: node_features
        1: edge_index
        2: edge_features
        3: node_targets
        4: node_positions

        Parameters
        ----------
        data : tuple[torch.Tensor, ...]
            Raw data tuple loaded from .pt file
        """
        node_features = data[0]  # [num_nodes][num_node_features]
        edge_index = data[1]  # [2][num_edges]
        edge_attr = data[2]  # [num_edges][num_edge_attributes]
        node_targets = data[3]  # [num_nodes][num_node_targets]
        node_positions = data[4]  # [num_nodes][2]
        return node_features, edge_index, edge_attr, node_targets, node_positions

    def _build_graph(self, data: tuple[torch.Tensor, ...]) -> Data:
        """
        Build PyG Data object from unpacked data.

        Parameters
        ----------
        data : tuple[torch.Tensor, ...]
            Raw data tuple loaded from .pt file

        Returns
        -------
        Data
            PyG Data object containing graph information.
        """
        node_features, edge_index, edge_attr, node_targets, node_positions = (
            self._unpack_data(data)
        )

        N, F = node_features.shape

        if N != node_targets.shape[0]:
            raise ValueError(
                f"Expected nodeFeatures to have shape ({node_targets.shape[0]}, {F}), but got ({N}, {F})"
            )
        if N != node_positions.shape[0]:
            raise ValueError(
                f"Expected nodeFeatures to have shape ({node_positions.shape[0]}, {F}), but got ({N}, {F})"
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                f"Expected edgeIndex to have shape (2, E), but got {edge_index.shape}"
            )

        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"Expected edgeAttr to have shape (E, num_edge_attributes), but got {edge_attr.shape}"
            )

        if self.use_edges:
            edge_index = reindex_edge_index(edge_index, torch.arange(N))
        else:
            edge_index = None
            edge_attr = None

        return Data(
            x=node_features.float(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_targets.long(),
            pos=node_positions.float(),
        )

    def len(self) -> int:
        """Return the number of data samples in the dataset."""
        return len(self.paths)

    def get(self, idx: int) -> Data:
        """Get the data sample at the specified index."""
        data = torch.load(self.paths[idx], weights_only=False)
        return self._build_graph(data)

    @property
    def ncols_(self) -> int:
        """Return number of columns in the NPS detector grid."""
        return NCOLS

    @property
    def nrows_(self) -> int:
        """Return number of rows in the NPS detector grid."""
        return NROWS

    @property
    def paths_(self) -> list[pathlib.Path]:
        """Return list of data file paths in the dataset."""
        return self.paths


class NPSDataLoader(BaseDataLoader):
    """DataLoader class for NPSDataset, wrapping around a NPSDataset instance."""

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        shuffle: bool = True,
        batch_size: int = 32,
        validation_split: float = 0.0,
        num_workers: int = 1,
        use_torch_loader: bool = False,
        **kwargs,
    ):
        """
        Initialize NPSDataLoader, which wraps around a NPSDataset instance.

        Parameters
        ----------
        dataset : Optional[Dataset]
            Pre-initialized dataset. If None, dataset will be initialized using data_paths, by default None
        logger : Optional[logging.Logger]
            logging for Dataset class
        shuffle : bool
            Whether to shuffle the dataset, by default True
        batch_size : int
            Size of each batch, by default 32
        validation_split : float
            Fraction of data to use for validation, by default 0.0
        num_workers : int
            Number of workers for data loading, by default 1
        """
        if dataset is None:
            dataset = NPSDataset(**kwargs)
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            use_torch_loader=use_torch_loader,
        )

    def __getattr__(self, name: str) -> any:
        """
        Dynamically forward attribute access to the underlying dataset. This allows accessing any dataset attribute/property through the dataloader.
        """
        try:
            return getattr(self.dataset, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object and its dataset have no attribute '{name}'"
            )
