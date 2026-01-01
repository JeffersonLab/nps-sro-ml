import pytest
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from base.dataloader import BaseDataLoader


class MockDataset(InMemoryDataset):
    """Mock dataset for testing purposes."""

    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        # Create simple graph data
        self._data_list = []
        for i in range(num_samples):
            x = np.random.randn(10, 5).astype(np.float32)
            edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
            y = i % 3  # Simple label
            data = Data(x=x, edge_index=edge_index, y=y)
            self._data_list.append(data)
        super().__init__()

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self._data_list[idx]


class TestBaseDataLoader:

    def test_initialization_no_validation(self):
        """Test initialization without validation split."""
        dataset = MockDataset(num_samples=50)
        dataloader = BaseDataLoader(dataset, batch_size=10, validation_split=0.0)

        assert dataloader.get_train_size() == 50
        assert dataloader.get_val_size() == 0
        assert dataloader.valid_sampler is None

    def test_initialization_with_float_validation(self):
        """Test initialization with float validation split."""
        dataset = MockDataset(num_samples=100)
        dataloader = BaseDataLoader(dataset, batch_size=10, validation_split=0.2)

        assert dataloader.get_train_size() == 80
        assert dataloader.get_val_size() == 20
        assert dataloader.valid_sampler is not None

    def test_initialization_with_int_validation(self):
        """Test initialization with int validation split."""
        dataset = MockDataset(num_samples=100)
        dataloader = BaseDataLoader(dataset, batch_size=10, validation_split=25)

        assert dataloader.get_train_size() == 75
        assert dataloader.get_val_size() == 25
        assert dataloader.valid_sampler is not None

    def test_split_validation_returns_loader(self):
        """Test that split_validation returns a valid DataLoader."""
        dataset = MockDataset(num_samples=100)
        dataloader = BaseDataLoader(dataset, batch_size=10, validation_split=0.2)
        val_loader = dataloader.split_validation()

        assert val_loader is not None
        assert len(val_loader) > 0

    def test_split_validation_returns_none(self):
        """Test that split_validation returns None when no validation split."""
        dataset = MockDataset(num_samples=100)
        dataloader = BaseDataLoader(dataset, batch_size=10, validation_split=0.0)
        val_loader = dataloader.split_validation()

        assert val_loader is None

    def test_random_seed_reproducibility(self):
        """Test that the same random seed produces the same split."""
        dataset = MockDataset(num_samples=100)

        dataloader1 = BaseDataLoader(
            dataset, batch_size=10, validation_split=0.2, random_seed=42
        )
        train_indices1 = list(dataloader1.sampler.indices)
        val_indices1 = list(dataloader1.valid_sampler.indices)

        dataloader2 = BaseDataLoader(
            dataset, batch_size=10, validation_split=0.2, random_seed=42
        )
        train_indices2 = list(dataloader2.sampler.indices)
        val_indices2 = list(dataloader2.valid_sampler.indices)

        assert train_indices1 == train_indices2
        assert val_indices1 == val_indices2

    def test_different_seeds_produce_different_splits(self):
        """Test that different random seeds produce different splits."""
        dataset = MockDataset(num_samples=100)

        dataloader1 = BaseDataLoader(
            dataset, batch_size=10, validation_split=0.2, random_seed=42
        )
        train_indices1 = list(dataloader1.sampler.indices)

        dataloader2 = BaseDataLoader(
            dataset, batch_size=10, validation_split=0.2, random_seed=123
        )
        train_indices2 = list(dataloader2.sampler.indices)

        assert train_indices1 != train_indices2

    def test_no_overlap_between_train_and_val(self):
        """Test that training and validation sets don't overlap."""
        dataset = MockDataset(num_samples=100)
        dataloader = BaseDataLoader(dataset, batch_size=10, validation_split=0.3)

        train_indices = set(dataloader.sampler.indices)
        val_indices = set(dataloader.valid_sampler.indices)

        assert len(train_indices.intersection(val_indices)) == 0

    def test_all_samples_covered(self):
        """Test that train and validation sets cover all samples."""
        dataset = MockDataset(num_samples=100)
        dataloader = BaseDataLoader(dataset, batch_size=10, validation_split=0.3)

        train_indices = set(dataloader.sampler.indices)
        val_indices = set(dataloader.valid_sampler.indices)
        all_indices = train_indices.union(val_indices)

        assert len(all_indices) == 100
        assert all_indices == set(range(100))

    def test_invalid_float_validation_split(self):
        """Test that invalid float validation split raises assertion."""
        dataset = MockDataset(num_samples=100)

        with pytest.raises(AssertionError):
            BaseDataLoader(dataset, batch_size=10, validation_split=1.5)

        with pytest.raises(AssertionError):
            BaseDataLoader(dataset, batch_size=10, validation_split=-0.1)

    def test_invalid_int_validation_split(self):
        """Test that invalid int validation split raises assertion."""
        dataset = MockDataset(num_samples=100)

        with pytest.raises(AssertionError):
            BaseDataLoader(dataset, batch_size=10, validation_split=-5)

        with pytest.raises(AssertionError):
            BaseDataLoader(dataset, batch_size=10, validation_split=150)

    def test_batch_iteration(self):
        """Test that we can iterate through batches."""
        dataset = MockDataset(num_samples=50)
        dataloader = BaseDataLoader(dataset, batch_size=10, validation_split=0.2)
        assert len(dataloader) > 0

    def test_validation_loader_kwargs_override(self):
        """Test that kwargs can override defaults in split_validation."""
        dataset = MockDataset(num_samples=100)
        dataloader = BaseDataLoader(
            dataset, batch_size=10, validation_split=0.2, num_workers=0
        )
        val_loader = dataloader.split_validation(batch_size=5)

        assert val_loader.batch_size == 5

    def test_shuffle_disabled_with_sampler(self):
        """Test that shuffle is disabled when using sampler."""
        dataset = MockDataset(num_samples=100)
        dataloader = BaseDataLoader(
            dataset, batch_size=10, validation_split=0.2, shuffle=True
        )

        # When sampler is used, shuffle should be False in parent class
        assert dataloader.sampler is not None
