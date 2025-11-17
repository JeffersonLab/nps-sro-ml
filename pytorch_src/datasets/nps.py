import pathlib
import torch
import logging
from typing import Optional
from torch_geometric.data import Dataset, Data
from base.dataloader import BaseDataLoader
from utils.utils import get_logger


NTIME = 110
NCOLS = 30
NROWS = 36

def get_node_index_from_position(row_idx, col_idx):
    return row_idx * NCOLS + col_idx


def get_position_from_node_index(node_idx):
    row_idx = node_idx // NCOLS
    col_idx = node_idx % NCOLS
    return row_idx, col_idx


class NPSDataset(Dataset):

    def __init__(
        self,
        paths: list[str|pathlib.Path],
        logger: Optional[logging.Logger] = None,
    ):
        """
        Constructor of NPSDataset, forces `transform`, `pre_transform`, and `pre_filter` to be None. If you want to apply any
        transformations or filtering, do it before initializing the dataset. Implementing those functionalities in this class lead to confusion. For instance, the `processed_file_names` is expected to be known beforehand, which is not possible if `pre_filter` skips some samples.
        """

        self.logger = get_logger("NPSDataset") if logger is None else logger

        paths = list(map(pathlib.Path, paths))
        self.paths = [pth for pth in paths if self._validate_file(pth)]

        broken = set(paths) - set(self.paths)
        for pth in broken:
            self.logger.warning(f"Invalid or broken data file skipped: {pth}")
        
        super().__init__(root="", transform=None, pre_transform=None, pre_filter=None)

    def _validate_file(self, path: pathlib.Path) -> bool:
        return path.suffix == ".pt" and path.is_file() and path.exists() and path.stat().st_size > 0

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        self.logger.info(f"Skip downloading as data are locally available at {self.paths[0].parent}.")

    def process(self):
        self.logger.info("Skip processing as data are pre-processed.")

    def _unpack_data(self, data):
        """
        Unpack raw data into components
        0: node_ids
        1: node_features
        2: node_targets
        3: edge_features
        4: edge_target_features
        5: edge_index
        6: edge_target_index
        """
        node_ids = data[0]      # [num_nodes]
        node_features = data[1] # [num_nodes][num_node_features]
        edge_index = data[6]    # [2][num_edges]
        return node_ids, node_features, edge_index

    def _build_graph(self, data):

        node_ids, node_features, edge_index = self._unpack_data(data)

        N, F = node_features.shape
        if N != node_ids.shape[0]:
            raise ValueError(
                f"Expected nodeFeatures to have shape ({node_ids.shape[0]}, {F}), but got ({N}, {F})"
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                f"Expected edgeIndex to have shape (2, E), but got {edge_index.shape}"
            )

        row, col = get_position_from_node_index(node_ids.to(torch.long))
        pos = torch.stack((row, col), dim=1).float()  # shape (N, 2)

        return Data(
            x=node_features.float(),  # node features (waveforms)
            edge_index=edge_index,  # edges will be defined dynamically later
            edge_attr=None,  # edge attributes not used
            y=None,  # node targets not used
            pos=pos,  # node positions according to the presence of waveforms
            time=None,  # time information is contained in nodeFeatures
        )

    def len(self): 
        return len(self.paths)

    def get(self, idx):
        data = torch.load(self.paths[idx], weights_only=False)
        return self._build_graph(data)

    @property
    def ncols_(self):
        return NCOLS

    @property
    def nrows_(self):
        return NROWS
    
    @property
    def paths_(self):
        return self.paths

class NPSDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset:Optional[Dataset]=None,
        data_paths:Optional[list[str|pathlib.Path]]=None,
        logger:Optional[logging.Logger]=None,
        shuffle: bool = True,
        batch_size: int = 32,
        validation_split: float = 0.0,
        num_workers: int = 1,
    ):
        """
        Constructor of NPSDataLoader.
        Parameters
        ----------  
        dataset : Optional[Dataset]
            Pre-initialized dataset. If None, dataset will be initialized using data_paths, by default None
        data_paths : Optional[list[str|pathlib.Path]]
            List of data file paths to initialize NPSDataset if dataset is None, by default None
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
        
        if dataset is None and data_paths is None:
            raise ValueError("Either 'dataset' or 'data_paths' must be provided.")
        
        if dataset is None:
            dataset = NPSDataset(paths=data_paths, logger=logger)

        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

    @property
    def num_features_(self):
        return self.dataset[0].num_features

    @property
    def num_classes_(self):
        return 2

    @property
    def ncols_(self):
        return self.dataset.ncols_
    
    @property
    def nrows_(self):
        return self.dataset.nrows_

    @property
    def paths_(self):
        return self.dataset.paths_