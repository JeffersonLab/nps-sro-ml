import torch
from utils.graph import create_unique_object_ids


class TestCreateUniqueObjectIds:
    """Unit tests for create_unique_object_ids function."""

    def test_basic_example_from_docstring(self):
        """Test the example provided in the docstring."""
        batch = torch.tensor([0, 0, 0, 1, 1, 2])
        object_ids = torch.tensor([0, 1, -1, 0, 1, -1])
        result = create_unique_object_ids(object_ids, batch, noise_idx=-1)
        expected = torch.tensor([0, 1, -1, 2, 3, -1])
        assert torch.equal(result, expected)

    def test_no_noise_nodes(self):
        """Test when there are no noise nodes."""
        batch = torch.tensor([0, 0, 1, 1])
        object_ids = torch.tensor([0, 1, 0, 1])
        result = create_unique_object_ids(object_ids, batch, noise_idx=-1)
        expected = torch.tensor([0, 1, 2, 3])
        assert torch.equal(result, expected)

    def test_all_noise_nodes(self):
        """Test when all nodes are noise."""
        batch = torch.tensor([0, 0, 1, 1])
        object_ids = torch.tensor([-1, -1, -1, -1])
        result = create_unique_object_ids(object_ids, batch, noise_idx=-1)
        expected = torch.tensor([-1, -1, -1, -1])
        assert torch.equal(result, expected)

    def test_single_batch(self):
        """Test with a single batch."""
        batch = torch.tensor([0, 0, 0])
        object_ids = torch.tensor([0, 1, -1])
        noise_idx = -1
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)
        expected = torch.tensor([0, 1, -1])
        assert torch.equal(result, expected)

    def test_preserves_noise_labels(self):
        """Test that noise labels are preserved."""
        batch = torch.tensor([0, 0, 1, 1, 2])
        object_ids = torch.tensor([0, -1, 0, -1, -1])
        noise_idx = -1
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)

        # All noise nodes should keep their label
        noise_mask = object_ids == noise_idx
        assert torch.all(result[noise_mask] == noise_idx)

    def test_unique_ids_per_batch_object_pair(self):
        """Test that same object ID in different batches gets unique IDs."""
        batch = torch.tensor([0, 0, 1, 1, 2, 2])
        object_ids = torch.tensor([5, 5, 5, 6, 5, 6])
        noise_idx = -1
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)

        # All nodes with (batch=0, oid=5) should have same new ID
        assert result[0] == result[1]
        # All nodes with (batch=1, oid=5) should have different ID than batch=0
        assert result[2] != result[0]
        # All nodes with (batch=2, oid=5) should have different ID than batch=0,1
        assert result[4] != result[0] and result[4] != result[2]

    def test_empty_tensors(self):
        """Test with empty tensors."""
        batch = torch.tensor([], dtype=torch.long)
        object_ids = torch.tensor([], dtype=torch.long)
        noise_idx = -1
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)
        assert result.shape == (0,)

    def test_single_node(self):
        """Test with a single node."""
        batch = torch.tensor([0])
        object_ids = torch.tensor([0])
        noise_idx = -1
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)
        expected = torch.tensor([0])
        assert torch.equal(result, expected)

    def test_device_compatibility(self):
        """Test that result is on the same device as input."""
        batch = torch.tensor([0, 0, 1, 1])
        object_ids = torch.tensor([0, 1, 0, 1])
        noise_idx = -1
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)
        assert result.device == object_ids.device

    def test_result_is_clone(self):
        """Test that result is a new tensor, not a reference."""
        batch = torch.tensor([0, 0, 1])
        object_ids = torch.tensor([0, -1, 0])
        noise_idx = -1
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)
        expected = torch.tensor([0, -1, 1])
        assert torch.equal(result, expected)

    def test_large_batch_indices(self):
        """Test with large batch indices."""
        batch = torch.tensor([0, 100, 100, 200])
        object_ids = torch.tensor([0, 0, 1, 0])
        noise_idx = -1
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)
        expect = torch.tensor([0, 1, 2, 3])
        assert torch.equal(result, expect)

    def test_non_consecutive_object_ids(self):
        """Test with non-consecutive object IDs."""
        batch = torch.tensor([0, 0, 1, 1, 2])
        object_ids = torch.tensor([10, 20, 10, 30, -1])
        noise_idx = -1
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)
        expect = torch.tensor([0, 1, 2, 3, -1])
        assert torch.equal(result, expect)

    def test_all_same_object_id(self):
        """Test when all nodes have the same object ID."""
        batch = torch.tensor([0, 0, 1, 1])
        object_ids = torch.tensor([5, 5, 5, 5])
        noise_idx = -1
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)
        expect = torch.tensor([0, 0, 1, 1])
        assert torch.equal(result, expect)

    def test_large_noise_idx(self):
        """Test with a large noise_idx."""
        batch = torch.tensor([0, 0, 1, 1, 1])
        object_ids = torch.tensor([0, 1, 0, 1, 999])
        noise_idx = 999
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)
        expected = torch.tensor([1000, 1001, 1002, 1003, 999])
        assert torch.equal(result, expected)

    def test_large_negative_noise_idx(self):
        """Test with a negative noise_idx."""
        batch = torch.tensor([0, 0, 1, 1, 1])
        object_ids = torch.tensor([0, 1, 0, 1, -999])
        noise_idx = -999
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)
        expected = torch.tensor([-998, -997, -996, -995, -999])
        assert torch.equal(result, expected)

    def test_randomized_input(self):
        """Test with randomized input."""
        torch.manual_seed(42)
        batch = torch.randint(0, 8, (256,))
        object_ids = torch.randint(-1, 1080, (256,))
        noise_idx = -1
        result = create_unique_object_ids(object_ids, batch, noise_idx=noise_idx)

        # Check that noise nodes are unchanged
        noise_mask = object_ids == noise_idx
        assert torch.all(result[noise_mask] == noise_idx)

        # Check that same (batch, oid) pairs have same new ID
        for b in batch.unique():
            for oid in object_ids[batch == b].unique():
                if oid == noise_idx:
                    continue
                mask = (batch == b) & (object_ids == oid)
                assert torch.all(result[mask] == result[mask][0])
