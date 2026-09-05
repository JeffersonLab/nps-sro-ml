import logging
import pathlib
from typing import Any, Optional, Protocol, Literal

import numpy as np
import torch

from base.dataloader import BaseDataLoader
from utils.graph import reindex_edge_index
from utils.utils import get_logger
from base.dataloader import Dataset, Data  # pyg if available, else torch

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


class _NPSDataSource(Protocol):
    paths: list[pathlib.Path]

    def len(self) -> int: ...
    def load(self, idx: int) -> Any: ...
    def to_tensors(self, raw: Any) -> dict[str, torch.Tensor]: ...

    @staticmethod
    def glob_paths(data_dir, max_files=None) -> list[pathlib.Path]: ...

    @staticmethod
    def validate_path(path: pathlib.Path) -> bool: ...

    def validate_data(self, tensors: dict[str, torch.Tensor]) -> Any:

        x = tensors["x"]
        edge_index = tensors["edge_index"]
        edge_attr = tensors["edge_attr"]
        pos = tensors["pos"]
        y = tensors["y"]

        if x.ndim != 2:
            raise ValueError(
                f"Expected features to have shape [num_nodes, num_features], got {x.shape}."
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"Expected edge index to have shape [2, num_edges], got {edge_index.shape}."
            )

        if edge_attr is not None and edge_attr.ndim != 2:
            raise ValueError(
                f"Expected edge attributes to have shape [num_edges, num_edge_attributes], got {edge_attr.shape}."
            )

        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(
                f"Expected geometry to have shape [num_nodes, 2], got {pos.shape}."
            )

        # if y.ndim != 1:
        #     raise ValueError(
        #         f"Expected targets to have shape [num_nodes], got {y.shape}."
        #     )

    @classmethod
    def from_data_dir(cls, data_dir, max_files=None, **kwargs):
        paths = cls.glob_paths(data_dir, max_files=max_files)
        paths = [p for p in paths if cls.validate_path(p)]
        return cls(paths=paths, **kwargs)


class _TorchSource(_NPSDataSource):
    def __init__(self, paths: list[pathlib.Path], max_files: Optional[int] = None):
        paths = list(filter(self.validate_path, paths))

        if max_files is not None and len(paths) > max_files:
            paths = paths[:max_files]
        self.paths = paths

    def len(self) -> int:
        return len(self.paths)

    def load(self, idx: int) -> tuple[torch.Tensor, ...]:
        return torch.load(self.paths[idx], weights_only=False)

    def to_tensors(self, raw: Any) -> dict[str, torch.Tensor]:
        num_nodes, _ = raw[0].shape
        edge_index = reindex_edge_index(raw[1], torch.arange(num_nodes))

        return {
            "x": raw[0],
            "edge_index": edge_index,
            "edge_attr": raw[2],
            "y": raw[3],
            "pos": raw[4],
        }

    @staticmethod
    def glob_paths(data_dir, max_files: Optional[int] = None) -> list[pathlib.Path]:
        paths = sorted(data_dir.glob("*.pt"))
        if max_files is not None:
            paths = paths[:max_files]
        return paths

    @staticmethod
    def validate_path(path: pathlib.Path) -> bool:
        return (
            path.suffix == ".pt"
            and path.is_file()
            and path.exists()
            and path.stat().st_size > 0
        )


class _NpySource(_NPSDataSource):

    REQUIRED_FILES = (
        "waveforms.npy",
        "hits.npy",
        "geometry.npy",
        "edge_index.npy",
        "cluster_index.npy",
        "cluster_type.npy",
    )

    def __init__(
        self,
        paths: list[pathlib.Path],
        max_files: Optional[int] = None,
        feature_mode: Literal["waveform", "hit"] = "waveform",
    ):
        self.paths = list(filter(self.validate_path, paths))
        if max_files is not None and len(self.paths) > max_files:
            self.paths = self.paths[:max_files]

        self.feature_mode = feature_mode

    def len(self) -> int:
        return len(self.paths)

    def load(self, idx: int) -> Any:
        event_dir = self.paths[idx]
        waveforms = np.load(event_dir / "waveforms.npy")
        hits = np.load(event_dir / "hits.npy")
        geometry = np.load(event_dir / "geometry.npy")
        edge_index = np.load(event_dir / "edge_index.npy")
        cluster_index = np.load(event_dir / "cluster_index.npy")
        cluster_type = np.load(event_dir / "cluster_type.npy")

        return waveforms, hits, geometry, edge_index, cluster_index, cluster_type

    def to_tensors(self, raw: Any) -> dict[str, torch.Tensor]:

        waveforms, hits, geometry, edge_index, cluster_index, cluster_type = raw

        pos = torch.as_tensor(geometry, dtype=torch.float32)

        if self.feature_mode == "waveform":
            x = torch.as_tensor(
                waveforms,
                dtype=torch.float32,
            )
        else:
            x = torch.as_tensor(hits, dtype=torch.float32)

        y = torch.as_tensor(cluster_index, dtype=torch.long)
        cluster_type = torch.as_tensor(cluster_type, dtype=torch.long)

        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        if edge_index.numel() == 0:
            edge_index = edge_index.reshape(2, 0)
        else:
            # To Do : include hit id during data generation to allow re-indexing of edge_index
            # hit_ids should have the same order as pos and y.
            # edge_index = reindex_edge_index(edge_index, hit_ids))
            pass

        data = {
            "x": x,  # Node features (either waveforms or hits)
            "edge_index": edge_index,
            "edge_attr": None,
            "y": y,  # Cluster indices for each node
            "pos": pos,
            "cluster_type": cluster_type,  # Type of each cluster
        }
        return data

    @staticmethod
    def glob_paths(
        data_dir: pathlib.Path, max_files: Optional[int] = None
    ) -> list[pathlib.Path]:
        paths = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        if max_files is not None:
            paths = paths[:max_files]
        return paths

    @staticmethod
    def validate_path(path: pathlib.Path) -> bool:
        if not path.is_dir():
            return False
        for filename in _NpySource.REQUIRED_FILES:
            file_path = path / filename
            if (
                not file_path.exists()
                or not file_path.is_file()
                or file_path.stat().st_size == 0
            ):
                return False
        return True


class NPSDataset(Dataset):
    def __init__(
        self,
        source: Literal["torch", "npy"] = "npy",
        paths: Optional[list[str | pathlib.Path]] = None,
        data_dir: Optional[pathlib.Path | str] = None,
        logger: Optional[logging.Logger] = None,
        max_files: Optional[int] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ):

        self.logger = get_logger("NPSDataset") if logger is None else logger
        if paths is None and data_dir is None:
            raise ValueError("Either 'paths' or 'data_dir' must be provided.")
        elif data_dir is not None:
            data_dir = pathlib.Path(data_dir)

        self.handler = self._build_handler(
            source, paths=paths, data_dir=data_dir, max_files=max_files, **kwargs
        )
        self.paths = self.handler.paths

        if len(self.paths) == 0:
            raise RuntimeError("No valid files found.")

        metadata_ = {
            "ncols": NCOLS,  # Number of columns in the detector grid
            "nrows": NROWS,  # Number of rows in the detector grid
            "nsamples": NTIME,  # Number of time samples in each waveform
        }
        if metadata is not None:
            metadata_.update(metadata)

        for key, value in metadata_.items():
            setattr(self, f"{key}_", value)

        root = self.paths[0].parent / ".pyg"
        super().__init__(root=root, transform=None, pre_transform=None, pre_filter=None)

    def _build_handler(
        self,
        source: Literal["torch", "npy"] = "npy",
        paths: Optional[list[str | pathlib.Path]] = None,
        data_dir: Optional[pathlib.Path | str] = None,
        max_files: Optional[int] = None,
        **kwargs,
    ):
        if paths is not None:
            paths = [pathlib.Path(p) for p in paths]

        if source == "torch":
            if paths is not None:
                handler = _TorchSource(paths=paths, max_files=max_files, **kwargs)
            elif data_dir is not None:
                handler = _TorchSource.from_data_dir(
                    data_dir, max_files=max_files, **kwargs
                )
        elif source == "npy":
            if paths is not None:
                handler = _NpySource(paths=paths, **kwargs)
            elif data_dir is not None:
                handler = _NpySource.from_data_dir(
                    data_dir, max_files=max_files, **kwargs
                )
        else:
            raise ValueError(f"Unknown source type: {source}")

        return handler

    def len(self):
        return self.handler.len()

    def get(self, idx):
        raw = self.handler.load(idx)
        tensors = self.handler.to_tensors(raw)
        self.handler.validate_data(tensors)
        return Data(**tensors)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        self.logger.info(
            f"Skip downloading as data are locally available at {self.paths[0].parent}."
        )

    def process(self):
        self.logger.info("Skip processing as data are pre-processed.")


class NPSDataLoader(BaseDataLoader):
    """DataLoader class for NPSDataset, wrapping around a NPSDataset instance."""

    def __init__(
        self,
        dataset: Optional[NPSDataset] = None,
        shuffle: bool = True,
        batch_size: int = 32,
        validation_split: float = 0.0,
        num_workers: int = 1,
        use_torch_loader: bool = False,
        **kwargs,
    ):
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
        try:
            return getattr(self.dataset, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object and its dataset have no attribute '{name}'"
            )
