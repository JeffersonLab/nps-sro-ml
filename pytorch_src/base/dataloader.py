import numpy as np
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Optional, Union


class BaseDataLoader(PygDataLoader):
    """
    Base DataLoader for PyTorch Geometric datasets with train/validation splitting.
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
    ):
        """
        Initialize the BaseDataLoader.
        Parameters
        ----------
        dataset : Dataset
            The PyTorch Geometric dataset to load data from.
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
        self.dataset = dataset

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        # Store init kwargs for creating validation loader
        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
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

    def split_validation(self, **kwargs) -> Optional[PygDataLoader]:
        """
        Create a validation DataLoader using the validation split.
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the PygDataLoader constructor. These will override the default initialization arguments.
        Returns
        -------
        Optional[PygDataLoader]
            A DataLoader for the validation set, or None if no validation split was defined.
        """

        if self.valid_sampler is None:
            return None

        # Create kwargs for validation loader
        val_kwargs = self.init_kwargs.copy()
        val_kwargs.update(kwargs)
        val_kwargs['shuffle'] = False

        # Validation loader uses the valid_sampler and never shuffles
        return PygDataLoader(sampler=self.valid_sampler, **val_kwargs)

    def get_train_size(self) -> int:
        """Return the number of training samples."""
        return self.n_samples

    def get_val_size(self) -> int:
        """Return the number of validation samples."""
        if self.valid_sampler is None:
            return 0
        return len(self.valid_sampler.indices)
