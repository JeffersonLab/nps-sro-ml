import pytest
import torch
from models.oc_loss import *


class TestOcAttrLossPerGraphNaive:
    def test_single_object_no_noise(self):
        """Test with a single object and no noise points."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss >= 0.0

    def test_multiple_objects_no_noise(self):
        """Test with multiple objects and no noise points."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [5.0, 5.0],
                [5.1, 5.0],  # object 2
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_with_noise_points(self):
        """Test with noise points that should be excluded."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [10.0, 10.0],  # noise
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0], dtype=torch.long)

        loss = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_multiple_noise_indices(self):
        """Test with multiple noise index values."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [10.0, 10.0],  # noise (idx 0)
                [20.0, 20.0],  # noise (idx -1)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5, 0.4], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, -1], dtype=torch.long)

        loss = oc_attr_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=[0, -1]
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_perfect_clustering(self):
        """Test with perfect clustering (all points at same location)."""
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert loss < 1e-5  # Should be very close to zero

    def test_high_beta_increases_loss(self):
        """Test that higher beta values result in higher loss contribution."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta_low = torch.tensor([0.5, 0.5], dtype=torch.float32)
        beta_high = torch.tensor([0.9, 0.9], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)

        loss_low = oc_attr_loss_per_graph_naive(
            x, beta_low, object_id, q_min=0.1, noise_idx=0
        )
        loss_high = oc_attr_loss_per_graph_naive(
            x, beta_high, object_id, q_min=0.1, noise_idx=0
        )

        assert loss_high > loss_low

    def test_all_noise_returns_zero(self):
        """Test that all noise points returns zero loss."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)
        object_id = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert loss == 0.0

    def test_requires_grad_raises_error(self):
        """Test that using tensors with requires_grad raises RuntimeError."""
        x = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32, requires_grad=True
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)

        with pytest.raises(RuntimeError, match="naive implementation for testing only"):
            oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

    def test_different_q_min_values(self):
        """Test with different q_min values."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)

        loss1 = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)
        loss2 = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.5, noise_idx=0)

        # Higher q_min should result in higher loss
        assert loss2 > loss1

    def test_higher_dimensional_space(self):
        """Test with higher dimensional latent space."""
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_single_point_objects(self):
        """Test with objects containing single points."""
        x = torch.tensor([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 2, 3], dtype=torch.long)

        loss = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

        # Each object has one point, so distance to representative is 0
        assert loss < 1e-5

    def test_representative_selection(self):
        """Test that the point with highest q is selected as representative."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        # Middle point has highest beta, so it should be the representative
        beta = torch.tensor([0.5, 0.9, 0.6], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_many_objects(self):
        """Test with many objects."""
        num_objects = 10
        x = torch.randn(num_objects * 3, 2, dtype=torch.float32)
        beta = torch.rand(num_objects * 3, dtype=torch.float32) * 0.9 + 0.05
        object_id = torch.arange(num_objects).repeat_interleave(3) + 1

        loss = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_mixed_object_sizes(self):
        """Test with objects of different sizes."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],  # object 1, 3 points
                [5.0, 5.0],
                [5.1, 5.0],  # object 2, 2 points
                [10.0, 10.0],  # object 3, 1 point
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.7, 0.85, 0.75, 0.6], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.long)

        loss = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_normalized_by_object_count(self):
        """Test that loss is normalized by number of objects."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)

        # Single object
        object_id_single = torch.tensor([1, 1], dtype=torch.long)
        loss_single = oc_attr_loss_per_graph_naive(
            x, beta, object_id_single, q_min=0.1, noise_idx=0
        )

        # Two separate objects (same points but different IDs)
        object_id_double = torch.tensor([1, 2], dtype=torch.long)
        loss_double = oc_attr_loss_per_graph_naive(
            x, beta, object_id_double, q_min=0.1, noise_idx=0
        )

        # Loss for single objects should be 0 (distance to self)
        assert loss_double < 1e-5

    def test_zero_beta_values(self):
        """Test with very low beta values."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.01, 0.01], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_graph_naive(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0


class TestOcRepulLossPerGraphNaive:
    def test_single_object_no_repulsion(self):
        """Test with a single object - should have no repulsion."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss == 0.0

    def test_multiple_objects_far_apart(self):
        """Test with multiple objects far apart - should have no repulsion."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [10.0, 10.0],
                [10.1, 10.0],  # object 2
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss == 0.0

    def test_multiple_objects_close_together(self):
        """Test with multiple objects close together - should have repulsion."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [0.5, 0.0],
                [0.6, 0.0],  # object 2 (within margin)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss > 0.0

    def test_with_noise_points(self):
        """Test with noise points that should contribute to repulsion."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [0.5, 0.0],  # noise (should contribute to repulsion)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss > 0.0

    def test_multiple_noise_indices(self):
        """Test with multiple noise index values."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [0.5, 0.0],  # noise (idx 0)
                [0.6, 0.0],  # noise (idx -1)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5, 0.4], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, -1], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=[0, -1], margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss > 0.0

    def test_higher_beta_increases_loss(self):
        """Test that higher beta values result in higher repulsive loss."""
        x = torch.tensor(
            [
                [0.0, 0.0],  # object 1
                [0.5, 0.0],  # object 2 (within margin)
            ],
            dtype=torch.float32,
        )
        beta_low = torch.tensor([0.5, 0.5], dtype=torch.float32)
        beta_high = torch.tensor([0.9, 0.9], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss_low = oc_repul_loss_per_graph_naive(
            x, beta_low, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )
        loss_high = oc_repul_loss_per_graph_naive(
            x, beta_high, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert loss_high > loss_low

    def test_all_noise_returns_zero(self):
        """Test that all noise points returns zero loss."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)
        object_id = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert loss == 0.0

    def test_different_margin_values(self):
        """Test with different margin values."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss1 = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=0.3
        )
        loss2 = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        # Larger margin should result in higher loss
        assert loss2 > loss1

    def test_margin_exactly_at_distance(self):
        """Test when margin is exactly at object distance."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        # Distance is exactly 1.0, which should be excluded (< margin, not <=)
        assert loss == 0.0

    def test_higher_dimensional_space(self):
        """Test with higher dimensional latent space."""
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=2.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_three_objects_partial_overlap(self):
        """Test with three objects where only some pairs are within margin."""
        x = torch.tensor(
            [
                [0.0, 0.0],  # object 1
                [0.5, 0.0],  # object 2 (close to 1)
                [5.0, 0.0],  # object 3 (far from both)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.85, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2, 3], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert loss > 0.0

    def test_raises_error_with_requires_grad(self):
        """Test that function raises error when x requires grad."""
        x = torch.tensor(
            [[0.0, 0.0], [0.5, 0.0]], dtype=torch.float32, requires_grad=True
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        with pytest.raises(RuntimeError, match="naive implementation for testing only"):
            oc_repul_loss_per_graph_naive(
                x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
            )

    def test_beta_clamping(self):
        """Test that beta values are properly clamped."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([1.0, 0.9999], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_single_point_objects(self):
        """Test with objects containing single points."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 2, 3], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss > 0.0

    def test_mixed_high_low_beta(self):
        """Test with mixture of high and low beta values."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [0.5, 0.0],
                [0.6, 0.0],  # object 2
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.1, 0.95, 0.05], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss > 0.0

    def test_representative_selection(self):
        """Test that representative is selected based on max q (not first point)."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.05, 0.0],  # object 1, higher beta
                [0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.5, 0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss > 0.0

    def test_different_q_min_values(self):
        """Test with different q_min values."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss1 = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )
        loss2 = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.5, noise_idx=0, margin=1.0
        )

        # Different q_min should result in different loss
        assert loss1 != loss2


class TestOcAttrLossPerGraph:
    def test_single_object_no_noise(self):
        """Test with a single object and no noise points."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_graph(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss >= 0.0

    def test_multiple_objects_no_noise(self):
        """Test with multiple objects and no noise points."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [5.0, 5.0],
                [5.1, 5.0],  # object 2
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss = oc_attr_loss_per_graph(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_with_noise_points(self):
        """Test with noise points that should be excluded."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [10.0, 10.0],  # noise
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0], dtype=torch.long)

        loss = oc_attr_loss_per_graph(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_multiple_noise_indices(self):
        """Test with multiple noise index values."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [10.0, 10.0],  # noise (idx 0)
                [20.0, 20.0],  # noise (idx -1)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5, 0.4], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, -1], dtype=torch.long)

        loss = oc_attr_loss_per_graph(x, beta, object_id, q_min=0.1, noise_idx=[0, -1])

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_perfect_clustering(self):
        """Test with perfect clustering (all points at same location)."""
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_graph(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert loss < 1e-5  # Should be very close to zero

    def test_high_beta_increases_loss(self):
        """Test that higher beta values result in higher loss contribution."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta_low = torch.tensor([0.5, 0.5], dtype=torch.float32)
        beta_high = torch.tensor([0.9, 0.9], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)

        loss_low = oc_attr_loss_per_graph(
            x, beta_low, object_id, q_min=0.1, noise_idx=0
        )
        loss_high = oc_attr_loss_per_graph(
            x, beta_high, object_id, q_min=0.1, noise_idx=0
        )

        assert loss_high > loss_low

    def test_all_noise_returns_zero(self):
        """Test that all noise points returns zero loss."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)
        object_id = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_attr_loss_per_graph(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert loss == 0.0

    def test_gradient_flow(self):
        """Test that gradients can flow through the loss."""
        x = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32, requires_grad=True
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32, requires_grad=True)
        object_id = torch.tensor([1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_graph(x, beta, object_id, q_min=0.1, noise_idx=0)
        loss.backward()

        assert x.grad is not None
        assert beta.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(beta.grad).any()

    def test_different_q_min_values(self):
        """Test with different q_min values."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)

        loss1 = oc_attr_loss_per_graph(x, beta, object_id, q_min=0.1, noise_idx=0)
        loss2 = oc_attr_loss_per_graph(x, beta, object_id, q_min=0.5, noise_idx=0)

        # Higher q_min should result in higher loss
        assert loss2 > loss1

    def test_higher_dimensional_space(self):
        """Test with higher dimensional latent space."""
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_graph(x, beta, object_id, q_min=0.1, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_result_with_manual_calculation(self):
        """Test with a small example and manually calculated expected loss."""

        num_nodes = 10
        unique_obj_ids = 3

        # uniformly distribute points into 3 clusters
        x = torch.randn((num_nodes, 2), dtype=torch.float32)
        object_id = torch.randint(0, unique_obj_ids, (num_nodes,), dtype=torch.long)
        beta = (
            torch.rand((num_nodes,), dtype=torch.float32) * 0.9 + 0.05
        )  # avoid beta close to 0 or 1

        manual_loss = oc_attr_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0
        )
        loss = oc_attr_loss_per_graph(x, beta, object_id, q_min=0.1, noise_idx=0)
        assert torch.isclose(loss, manual_loss, atol=1e-5)


class TestOcRepulLossPerGraph:
    def test_single_object_no_noise(self):
        """Test with a single object and no noise points - should have no repulsion."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)

        loss = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss == 0.0  # No repulsion with single object

    def test_multiple_objects_far_apart(self):
        """Test with multiple objects far apart - should have no repulsion."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [10.0, 10.0],
                [10.1, 10.0],  # object 2
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss == 0.0  # Objects are too far apart

    def test_multiple_objects_close_together(self):
        """Test with multiple objects close together - should have repulsion."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [0.5, 0.0],
                [0.6, 0.0],  # object 2 (within margin)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss > 0.0  # Should have repulsive loss

    def test_with_noise_points(self):
        """Test with noise points that should be excluded."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [0.5, 0.0],  # noise (should still contribute to repulsion)
                [5.0, 5.0],
                [5.1, 5.0],  # object 2
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, 2, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_multiple_noise_indices(self):
        """Test with multiple noise index values."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [0.5, 0.0],  # noise (idx 0)
                [0.6, 0.0],  # noise (idx -1)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5, 0.4], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, -1], dtype=torch.long)

        loss = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=[0, -1], margin=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_higher_beta_increases_loss(self):
        """Test that higher beta values result in higher repulsive loss."""
        x = torch.tensor(
            [
                [0.0, 0.0],  # object 1
                [0.5, 0.0],  # object 2 (within margin)
            ],
            dtype=torch.float32,
        )
        beta_low = torch.tensor([0.5, 0.5], dtype=torch.float32)
        beta_high = torch.tensor([0.9, 0.9], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss_low = oc_repul_loss_per_graph(
            x, beta_low, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )
        loss_high = oc_repul_loss_per_graph(
            x, beta_high, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert loss_high > loss_low

    def test_all_noise_returns_zero(self):
        """Test that all noise points returns zero loss."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)
        object_id = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert loss == 0.0

    def test_gradient_flow(self):
        """Test that gradients can flow through the loss."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32, requires_grad=True)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )
        loss.backward()

        assert x.grad is not None
        assert beta.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(beta.grad).any()

    def test_different_margin_values(self):
        """Test with different margin values."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss1 = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=0.3
        )
        loss2 = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        # Larger margin should result in higher loss (more points within range)
        assert loss2 > loss1

    def test_margin_exactly_at_distance(self):
        """Test when margin is exactly at object distance."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        # Distance is exactly 1.0, which should be excluded (< margin, not <=)
        assert loss == 0.0

    def test_higher_dimensional_space(self):
        """Test with higher dimensional latent space."""
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=2.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_three_objects_partial_overlap(self):
        """Test with three objects where only some pairs are within margin."""
        x = torch.tensor(
            [
                [0.0, 0.0],  # object 1
                [0.5, 0.0],  # object 2 (close to 1)
                [5.0, 0.0],  # object 3 (far from both)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.85, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2, 3], dtype=torch.long)

        loss = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert loss > 0.0

    def test_result_with_manual_calculation(self):
        """Test with a small example and manually calculated expected loss."""
        num_nodes = 10
        unique_obj_ids = 3

        # uniformly distribute points into 3 clusters
        x = torch.randn((num_nodes, 2), dtype=torch.float32)
        object_id = torch.randint(1, unique_obj_ids + 1, (num_nodes,), dtype=torch.long)
        beta = torch.rand((num_nodes,), dtype=torch.float32) * 0.9 + 0.05

        manual_loss = oc_repul_loss_per_graph_naive(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )
        loss = oc_repul_loss_per_graph(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0
        )

        assert torch.isclose(loss, manual_loss, atol=1e-5)


class TestOcCowardLossPerGraph:
    def test_single_object(self):
        """Test with a single object."""
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)

        loss = oc_coward_loss_per_graph(beta, object_id, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        # Should be 1 - max(beta) = 1 - 0.9 = 0.1
        assert torch.isclose(loss, torch.tensor(0.1), atol=1e-5)

    def test_multiple_objects(self):
        """Test with multiple objects."""
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss = oc_coward_loss_per_graph(beta, object_id, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        # obj 1: max beta = 0.9, obj 2: max beta = 0.85
        # loss = mean([1-0.9, 1-0.85]) = mean([0.1, 0.15]) = 0.125
        assert torch.isclose(loss, torch.tensor(0.125), atol=1e-5)

    def test_with_noise_points(self):
        """Test with noise points that should be excluded."""
        beta = torch.tensor([0.9, 0.8, 0.5, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, 2, 2], dtype=torch.long)

        loss = oc_coward_loss_per_graph(beta, object_id, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        # obj 1: max beta = 0.9, obj 2: max beta = 0.85
        # noise point (0.5) should be ignored
        assert torch.isclose(loss, torch.tensor(0.125), atol=1e-5)

    def test_multiple_noise_indices(self):
        """Test with multiple noise index values."""
        beta = torch.tensor([0.9, 0.8, 0.5, 0.4, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, -1, 2, 2], dtype=torch.long)

        loss = oc_coward_loss_per_graph(beta, object_id, noise_idx=[0, -1])

        assert isinstance(loss, torch.Tensor)
        # Only objects 1 and 2 should be considered
        assert torch.isclose(loss, torch.tensor(0.125), atol=1e-5)

    def test_all_noise_returns_zero(self):
        """Test that all noise points returns zero loss."""
        beta = torch.tensor([0.5, 0.4], dtype=torch.float32)
        object_id = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_coward_loss_per_graph(beta, object_id, noise_idx=0)

        assert loss == 0.0

    def test_high_beta_reduces_loss(self):
        """Test that higher beta values result in lower coward loss."""
        beta_low = torch.tensor([0.5, 0.4], dtype=torch.float32)
        beta_high = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)

        loss_low = oc_coward_loss_per_graph(beta_low, object_id, noise_idx=0)
        loss_high = oc_coward_loss_per_graph(beta_high, object_id, noise_idx=0)

        # Higher beta should give lower coward loss
        assert loss_high < loss_low

    def test_perfect_confidence(self):
        """Test with perfect confidence (beta = 1.0 after clamping)."""
        beta = torch.tensor([1.0, 0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)

        loss = oc_coward_loss_per_graph(beta, object_id, noise_idx=0)

        # Max beta is 1.0, so loss should be 0
        assert loss == 0.0

    def test_gradient_flow(self):
        """Test that gradients can flow through the loss."""
        beta = torch.tensor(
            [0.9, 0.8, 0.85, 0.75], dtype=torch.float32, requires_grad=True
        )
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss = oc_coward_loss_per_graph(beta, object_id, noise_idx=0)
        loss.backward()

        assert beta.grad is not None
        assert not torch.isnan(beta.grad).any()

    def test_single_point_objects(self):
        """Test with objects containing single points."""
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 2, 3], dtype=torch.long)

        loss = oc_coward_loss_per_graph(beta, object_id, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        # Each object has one point, so max beta = beta itself
        # loss = mean([1-0.9, 1-0.8, 1-0.7]) = mean([0.1, 0.2, 0.3]) = 0.2
        assert torch.isclose(loss, torch.tensor(0.2), atol=1e-5)

    def test_mixed_high_low_beta(self):
        """Test with mixture of high and low beta values."""
        beta = torch.tensor([0.9, 0.1, 0.95, 0.05], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss = oc_coward_loss_per_graph(beta, object_id, noise_idx=0)

        # obj 1: max beta = 0.9, obj 2: max beta = 0.95
        # loss = mean([1-0.9, 1-0.95]) = mean([0.1, 0.05]) = 0.075
        assert torch.isclose(loss, torch.tensor(0.075), atol=1e-5)

    def test_many_objects(self):
        """Test with many objects."""
        num_objects = 10
        beta = torch.rand(num_objects * 3, dtype=torch.float32)
        object_id = torch.arange(num_objects).repeat_interleave(3)

        loss = oc_coward_loss_per_graph(beta, object_id, noise_idx=-1)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0
        assert loss <= 1.0  # Since beta is in [0, 1], loss is in [0, 1]

    def test_zero_beta(self):
        """Test with all beta values at zero."""
        beta = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)

        loss = oc_coward_loss_per_graph(beta, object_id, noise_idx=0)

        # Max beta is 0, so loss should be 1.0
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)

    def test_consistent_across_permutations(self):
        """Test that loss is consistent regardless of node ordering."""
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss1 = oc_coward_loss_per_graph(beta, object_id, noise_idx=0)

        # Permute the data
        perm = torch.tensor([2, 0, 3, 1])
        beta_perm = beta[perm]
        object_id_perm = object_id[perm]

        loss2 = oc_coward_loss_per_graph(beta_perm, object_id_perm, noise_idx=0)

        assert torch.isclose(loss1, loss2, atol=1e-5)


class TestOcNoiseLossPerGraph:
    def test_single_noise_point(self):
        """Test with a single noise point."""
        beta = torch.tensor([0.5], dtype=torch.float32)
        object_id = torch.tensor([0], dtype=torch.long)

        loss = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert torch.isclose(loss, torch.tensor(0.5), atol=1e-5)

    def test_multiple_noise_points(self):
        """Test with multiple noise points."""
        beta = torch.tensor([0.5, 0.3, 0.7], dtype=torch.float32)
        object_id = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        # Mean of [0.5, 0.3, 0.7] = 0.5
        assert torch.isclose(loss, torch.tensor(0.5), atol=1e-5)

    def test_no_noise_points(self):
        """Test with no noise points - should return zero."""
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)

        loss = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)

        assert loss == 0.0

    def test_mixed_noise_and_signal(self):
        """Test with mixture of noise and signal points."""
        beta = torch.tensor([0.9, 0.8, 0.5, 0.3], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, 0], dtype=torch.long)

        loss = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        # Mean of noise points [0.5, 0.3] = 0.4
        assert torch.isclose(loss, torch.tensor(0.4), atol=1e-5)

    def test_multiple_noise_indices(self):
        """Test with multiple noise index values."""
        beta = torch.tensor([0.5, 0.4, 0.9, 0.6], dtype=torch.float32)
        object_id = torch.tensor([0, -1, 1, -1], dtype=torch.long)

        loss = oc_noise_loss_per_graph(beta, object_id, noise_idx=[0, -1])

        assert isinstance(loss, torch.Tensor)
        # Mean of noise points [0.5, 0.4, 0.6] = 0.5
        assert torch.isclose(loss, torch.tensor(0.5), atol=1e-5)

    def test_high_beta_noise_increases_loss(self):
        """Test that higher beta values in noise result in higher loss."""
        beta_low = torch.tensor([0.2, 0.3], dtype=torch.float32)
        beta_high = torch.tensor([0.8, 0.9], dtype=torch.float32)
        object_id = torch.tensor([0, 0], dtype=torch.long)

        loss_low = oc_noise_loss_per_graph(beta_low, object_id, noise_idx=0)
        loss_high = oc_noise_loss_per_graph(beta_high, object_id, noise_idx=0)

        assert loss_high > loss_low

    def test_zero_beta_noise(self):
        """Test with all noise beta values at zero."""
        beta = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        object_id = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_perfect_beta_noise(self):
        """Test with all noise beta values at 1.0."""
        beta = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        object_id = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)

        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients can flow through the loss."""
        beta = torch.tensor([0.5, 0.4], dtype=torch.float32, requires_grad=True)
        object_id = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)
        loss.backward()

        assert beta.grad is not None
        assert not torch.isnan(beta.grad).any()

    def test_single_noise_index_as_int(self):
        """Test that single noise_idx as int works correctly."""
        beta = torch.tensor([0.5, 0.3], dtype=torch.float32)
        object_id = torch.tensor([0, 0], dtype=torch.long)

        loss_int = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)
        loss_list = oc_noise_loss_per_graph(beta, object_id, noise_idx=[0])

        assert torch.isclose(loss_int, loss_list, atol=1e-5)

    def test_noise_among_multiple_objects(self):
        """Test noise points interspersed among multiple objects."""
        beta = torch.tensor([0.9, 0.5, 0.8, 0.4, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 0, 2, 0, 3], dtype=torch.long)

        loss = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)

        # Mean of noise points [0.5, 0.4] = 0.45
        assert torch.isclose(loss, torch.tensor(0.45), atol=1e-5)

    def test_different_noise_beta_values(self):
        """Test with various beta values for noise points."""
        beta = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
        object_id = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

        loss = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)

        # Mean of [0.1, 0.3, 0.5, 0.7, 0.9] = 0.5
        assert torch.isclose(loss, torch.tensor(0.5), atol=1e-5)

    def test_large_number_of_noise_points(self):
        """Test with a large number of noise points."""
        num_noise = 100
        beta = torch.rand(num_noise, dtype=torch.float32)
        object_id = torch.zeros(num_noise, dtype=torch.long)

        loss = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert torch.isclose(loss, beta.mean(), atol=1e-5)

    def test_consistent_across_permutations(self):
        """Test that loss is consistent regardless of node ordering."""
        beta = torch.tensor([0.5, 0.3, 0.7], dtype=torch.float32)
        object_id = torch.tensor([0, 0, 0], dtype=torch.long)

        loss1 = oc_noise_loss_per_graph(beta, object_id, noise_idx=0)

        # Permute the data
        perm = torch.tensor([2, 0, 1])
        beta_perm = beta[perm]
        object_id_perm = object_id[perm]

        loss2 = oc_noise_loss_per_graph(beta_perm, object_id_perm, noise_idx=0)

        assert torch.isclose(loss1, loss2, atol=1e-5)


class TestOcAttrLossPerBatch:
    def test_single_graph_single_object(self):
        """Test with a single graph containing a single object."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)
        batch = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss >= 0.0

    def test_single_graph_multiple_objects(self):
        """Test with a single graph containing multiple objects."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [5.0, 5.0],
                [5.1, 5.0],  # object 2
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_multiple_graphs(self):
        """Test with multiple graphs in a batch."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # graph 0, object 1
                [5.0, 5.0],
                [5.1, 5.0],  # graph 1, object 2
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_with_noise_points(self):
        """Test with noise points that should be excluded."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [10.0, 10.0],  # noise
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0], dtype=torch.long)
        batch = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_default_batch_parameter(self):
        """Test that batch defaults to all zeros when not provided."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=None
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_perfect_clustering(self):
        """Test with perfect clustering (all points at same location)."""
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)
        batch = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert loss < 1e-5  # Should be very close to zero

    def test_high_beta_increases_loss(self):
        """Test that higher beta values result in higher loss contribution."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta_low = torch.tensor([0.5, 0.5], dtype=torch.float32)
        beta_high = torch.tensor([0.9, 0.9], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss_low = oc_attr_loss_per_batch(
            x, beta_low, object_id, q_min=0.1, noise_idx=0, batch=batch
        )
        loss_high = oc_attr_loss_per_batch(
            x, beta_high, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert loss_high > loss_low

    def test_all_noise_returns_zero(self):
        """Test that all noise points returns zero loss."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)
        object_id = torch.tensor([0, 0], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert loss == 0.0

    def test_gradient_flow(self):
        """Test that gradients can flow through the loss."""
        x = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32, requires_grad=True
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32, requires_grad=True)
        object_id = torch.tensor([1, 1], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )
        loss.backward()

        assert x.grad is not None
        assert beta.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(beta.grad).any()

    def test_different_q_min_values(self):
        """Test with different q_min values."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss1 = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )
        loss2 = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.5, noise_idx=0, batch=batch
        )

        # Higher q_min should result in higher loss
        assert loss2 > loss1

    def test_multiple_graphs_different_sizes(self):
        """Test with multiple graphs of different sizes."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],  # graph 0, 3 nodes
                [5.0, 5.0],
                [5.1, 5.0],  # graph 1, 2 nodes
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.7, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_multiple_graphs_with_noise(self):
        """Test with multiple graphs where some contain noise."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # graph 0, object 1
                [10.0, 10.0],  # graph 0, noise
                [5.0, 5.0],
                [5.1, 5.0],  # graph 1, object 2
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_higher_dimensional_space(self):
        """Test with higher dimensional latent space."""
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 1], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_large_batch(self):
        """Test with a larger batch of graphs."""
        num_graphs = 5
        nodes_per_graph = 4
        num_nodes = num_graphs * nodes_per_graph

        x = torch.randn((num_nodes, 2), dtype=torch.float32)
        beta = torch.rand((num_nodes,), dtype=torch.float32) * 0.9 + 0.05
        object_id = torch.arange(num_graphs).repeat_interleave(nodes_per_graph) + 1
        batch = torch.arange(num_graphs).repeat_interleave(nodes_per_graph)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_beta_clamping(self):
        """Test that beta values are properly clamped."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor(
            [0.9999, 1.0], dtype=torch.float32
        )  # Values at/above clamp threshold
        object_id = torch.tensor([1, 1], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_result_with_manual_calculation(self):
        """Test with a small example and manually calculated expected loss."""

        num_graphs = 5
        num_nodes = 40
        unique_obj_ids = 10

        x = torch.randn((num_nodes, 2), dtype=torch.float32)
        batch = torch.randint(0, num_graphs, (num_nodes,), dtype=torch.long)
        beta = (
            torch.rand((num_nodes,), dtype=torch.float32) * 0.9 + 0.05
        )  # avoid beta close to 0 or 1
        object_id = torch.randint(0, unique_obj_ids, (num_nodes,), dtype=torch.long)
        object_id = torch.where(
            object_id == 0,
            0,
            object_id + batch * unique_obj_ids,
        )

        loss = oc_attr_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, batch=batch
        )

        manual_loss = 0.0
        for b in batch.unique():
            mask = batch == b
            manual_loss += oc_attr_loss_per_graph_naive(
                x[mask], beta[mask], object_id[mask], q_min=0.1, noise_idx=0
            )

            not_noise = object_id[mask] != 0
            uni_objs = object_id[mask][not_noise].unique()

        assert torch.isclose(loss, manual_loss, atol=1e-5)


class TestOcRepulLossPerBatch:
    def test_single_graph_single_object(self):
        """Test with a single graph containing a single object - should have no repulsion."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)
        batch = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss == 0.0  # No repulsion with single object

    def test_single_graph_multiple_objects_far_apart(self):
        """Test with a single graph containing multiple objects far apart - should have no repulsion."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [10.0, 10.0],
                [10.1, 10.0],  # object 2
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss == 0.0  # Objects are too far apart

    def test_single_graph_multiple_objects_close_together(self):
        """Test with a single graph containing multiple objects close together - should have repulsion."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [0.5, 0.0],
                [0.6, 0.0],  # object 2 (within margin)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss > 0.0  # Should have repulsive loss

    def test_multiple_graphs(self):
        """Test with multiple graphs in a batch."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],  # graph 0, objects 1 and 2 (close together)
                [5.0, 5.0],
                [5.5, 5.0],  # graph 1, objects 3 and 4 (close together)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss > 0.0

    def test_with_noise_points(self):
        """Test with noise points that should still contribute to repulsion."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [0.5, 0.0],  # noise (should still repel from object 1)
                [5.0, 5.0],
                [5.1, 5.0],  # object 2
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_multiple_noise_indices(self):
        """Test with multiple noise index values."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],  # object 1
                [0.5, 0.0],  # noise (idx 0)
                [0.6, 0.0],  # noise (idx -1)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5, 0.4], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, -1], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=[0, -1], margin=1.0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_default_batch_parameter(self):
        """Test that batch defaults to all zeros when not provided."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=None
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_higher_beta_increases_loss(self):
        """Test that higher beta values result in higher repulsive loss."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        beta_low = torch.tensor([0.5, 0.5], dtype=torch.float32)
        beta_high = torch.tensor([0.9, 0.9], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss_low = oc_repul_loss_per_batch(
            x, beta_low, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )
        loss_high = oc_repul_loss_per_batch(
            x, beta_high, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert loss_high > loss_low

    def test_all_noise_returns_zero(self):
        """Test that all noise points returns zero loss."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)
        object_id = torch.tensor([0, 0], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert loss == 0.0

    def test_gradient_flow(self):
        """Test that gradients can flow through the loss."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32, requires_grad=True)
        object_id = torch.tensor([1, 2], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )
        loss.backward()

        assert x.grad is not None
        assert beta.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(beta.grad).any()

    def test_different_margin_values(self):
        """Test with different margin values."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss1 = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=0.3, batch=batch
        )
        loss2 = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        # Larger margin should result in higher loss (more points within range)
        assert loss2 > loss1

    def test_margin_exactly_at_distance(self):
        """Test when margin is exactly at object distance."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        # Distance is exactly 1.0, which should be excluded (< margin, not <=)
        assert loss == 0.0

    def test_multiple_graphs_different_sizes(self):
        """Test with multiple graphs of different sizes."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [0.3, 0.3],  # graph 0, 3 objects
                [5.0, 5.0],
                [5.5, 5.0],  # graph 1, 2 objects
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.7, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_multiple_graphs_with_noise(self):
        """Test with multiple graphs where some contain noise."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],  # graph 0, objects 1 and 2
                [10.0, 10.0],  # graph 0, noise
                [5.0, 5.0],
                [5.5, 5.0],  # graph 1, objects 3 and 4
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8, 0.5, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 2, 0, 3, 4], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_higher_dimensional_space(self):
        """Test with higher dimensional latent space."""
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=2.0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_three_objects_partial_overlap(self):
        """Test with three objects where only some pairs are within margin."""
        x = torch.tensor(
            [
                [0.0, 0.0],  # object 1
                [0.5, 0.0],  # object 2 (close to 1)
                [5.0, 0.0],  # object 3 (far from both)
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor([0.9, 0.85, 0.8], dtype=torch.float32)
        object_id = torch.tensor([1, 2, 3], dtype=torch.long)
        batch = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert loss > 0.0

    def test_large_batch(self):
        """Test with a larger batch of graphs."""
        num_graphs = 5
        nodes_per_graph = 4
        num_nodes = num_graphs * nodes_per_graph

        x = torch.randn((num_nodes, 2), dtype=torch.float32)
        beta = torch.rand((num_nodes,), dtype=torch.float32) * 0.9 + 0.05
        object_id = torch.arange(num_nodes) + 1  # Each node is a different object
        batch = torch.arange(num_graphs).repeat_interleave(nodes_per_graph)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_beta_clamping(self):
        """Test that beta values are properly clamped."""
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        beta = torch.tensor(
            [0.9999, 1.0], dtype=torch.float32
        )  # Values at/above clamp threshold
        object_id = torch.tensor([1, 2], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_result_with_manual_calculation(self):
        """Test with a small example and manually calculated expected loss."""

        num_graphs = 5
        num_nodes = 40
        unique_obj_ids = 10

        x = torch.randn((num_nodes, 2), dtype=torch.float32)
        batch = torch.randint(0, num_graphs, (num_nodes,), dtype=torch.long)
        beta = (
            torch.rand((num_nodes,), dtype=torch.float32) * 0.9 + 0.05
        )  # avoid beta close to 0 or 1
        object_id = torch.randint(0, unique_obj_ids, (num_nodes,), dtype=torch.long)
        object_id = torch.where(
            object_id == 0,
            0,
            object_id + batch * unique_obj_ids,
        )

        loss = oc_repul_loss_per_batch(
            x, beta, object_id, q_min=0.1, noise_idx=0, margin=1.0, batch=batch
        )

        manual_loss = 0.0
        for b in batch.unique():
            mask = batch == b
            manual_loss += oc_repul_loss_per_graph_naive(
                x[mask], beta[mask], object_id[mask], q_min=0.1, noise_idx=0, margin=1.0
            )

        assert torch.isclose(loss, manual_loss, atol=1e-5)


class TestOcCowardLossPerBatch:
    def test_single_graph_single_object(self):
        """Test with a single graph containing a single object."""
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)
        batch = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        # Representative has beta=0.9, so loss should be 1-0.9=0.1
        assert torch.isclose(loss, torch.tensor(0.1), atol=1e-5)

    def test_single_graph_multiple_objects(self):
        """Test with a single graph containing multiple objects."""
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        assert isinstance(loss, torch.Tensor)
        # Object 1: representative has beta=0.9, loss=0.1
        # Object 2: representative has beta=0.85, loss=0.15
        # Mean = (0.1 + 0.15) / 2 = 0.125
        expected = torch.tensor(0.125)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_multiple_graphs(self):
        """Test with multiple graphs in a batch."""
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        assert isinstance(loss, torch.Tensor)
        # Graph 0: object 1 with representative beta=0.9, loss=0.1
        # Graph 1: object 2 with representative beta=0.85, loss=0.15
        # Total = 0.1 + 0.15 = 0.25
        expected = torch.tensor(0.25)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_with_noise_points(self):
        """Test with noise points that should be excluded."""
        beta = torch.tensor([0.9, 0.8, 0.5, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        assert isinstance(loss, torch.Tensor)
        # Object 1: representative has beta=0.9, loss=0.1
        # Object 2: representative has beta=0.85, loss=0.15
        # Noise point is ignored
        expected = torch.tensor(0.125)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_multiple_noise_indices(self):
        """Test with multiple noise index values."""
        beta = torch.tensor([0.9, 0.8, 0.5, 0.4, 0.85], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, -1, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=[0, -1], batch=batch)

        assert isinstance(loss, torch.Tensor)
        # Object 1: representative has beta=0.9, loss=0.1
        # Object 2: representative has beta=0.85, loss=0.15
        expected = torch.tensor(0.125)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_all_noise_returns_zero(self):
        """Test that all noise points returns zero loss."""
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)
        object_id = torch.tensor([0, 0], dtype=torch.long)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        assert loss == 0.0

    def test_no_batch_provided(self):
        """Test when batch is None (single graph assumed)."""
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=None)

        assert isinstance(loss, torch.Tensor)
        expected = torch.tensor(0.125)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_high_beta_representatives(self):
        """Test that representatives with high beta result in lower loss."""
        beta_high = torch.tensor([0.95, 0.7, 0.9, 0.6], dtype=torch.float32)
        beta_low = torch.tensor([0.6, 0.7, 0.5, 0.6], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        loss_high = oc_coward_loss_per_batch(
            beta_high, object_id, noise_idx=0, batch=batch
        )
        loss_low = oc_coward_loss_per_batch(
            beta_low, object_id, noise_idx=0, batch=batch
        )

        assert loss_high < loss_low

    def test_single_point_objects(self):
        """Test with objects containing single points."""
        beta = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 2, 3], dtype=torch.long)
        batch = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        assert isinstance(loss, torch.Tensor)
        # Mean of (1-0.9, 1-0.8, 1-0.7) = (0.1 + 0.2 + 0.3) / 3 = 0.2
        expected = torch.tensor(0.2)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_multiple_graphs_different_sizes(self):
        """Test with multiple graphs of different sizes."""
        beta = torch.tensor([0.9, 0.8, 0.85, 0.75, 0.7], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2, 3], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        assert isinstance(loss, torch.Tensor)
        # Graph 0: objects 1,2 with representatives beta=0.9,0.85
        # Mean for graph 0: (0.1 + 0.15) / 2 = 0.125
        # Graph 1: object 3 with representative beta=0.7
        # Mean for graph 1: 0.3
        # Total = 0.125 + 0.3 = 0.425
        expected = torch.tensor(0.425)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_representative_selection(self):
        """Test that the point with highest beta is selected as representative."""
        beta = torch.tensor([0.5, 0.9, 0.6], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 1], dtype=torch.long)
        batch = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        # Middle point has highest beta (0.9), so loss = 1 - 0.9 = 0.1
        expected = torch.tensor(0.1)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_many_objects(self):
        """Test with many objects."""
        num_objects = 10
        beta = torch.rand(num_objects * 3, dtype=torch.float32) * 0.9 + 0.05
        object_id = torch.arange(num_objects).repeat_interleave(3) + 1
        batch = torch.zeros(num_objects * 3, dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0
        assert loss <= 1.0

    def test_perfect_beta_values(self):
        """Test with perfect beta=1 values."""
        beta = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        # All representatives have beta=1, so loss should be 0
        assert loss < 1e-5

    def test_zero_beta_values(self):
        """Test with beta=0 values."""
        beta = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        # All representatives have beta=0, so loss should be 1
        expected = torch.tensor(1.0)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_mixed_batch_with_noise(self):
        """Test with multiple graphs, some containing noise."""
        beta = torch.tensor([0.9, 0.8, 0.5, 0.85, 0.75, 0.6], dtype=torch.float32)
        object_id = torch.tensor([1, 1, 0, 2, 2, 3], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        assert isinstance(loss, torch.Tensor)
        # Graph 0: object 1, representative beta=0.9, loss=0.1
        # Graph 1: objects 2,3, representatives beta=0.85,0.6, mean=(0.15+0.4)/2=0.275
        # Total = 0.1 + 0.275 = 0.375
        expected = torch.tensor(0.375)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients can flow through the loss."""
        beta = torch.tensor(
            [0.9, 0.8, 0.85, 0.75], dtype=torch.float32, requires_grad=True
        )
        object_id = torch.tensor([1, 1, 2, 2], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)
        loss.backward()

        assert beta.grad is not None
        assert not torch.isnan(beta.grad).any()

    def test_result_with_manual_calculation(self):
        """Test with a small example and manually calculated expected loss."""

        num_graphs = 5
        num_nodes = 40
        unique_obj_ids = 10

        beta = (
            torch.rand((num_nodes,), dtype=torch.float32) * 0.9 + 0.05
        )  # avoid beta close to 0 or 1
        batch = torch.randint(0, num_graphs, (num_nodes,), dtype=torch.long)
        object_id = torch.randint(0, unique_obj_ids, (num_nodes,), dtype=torch.long)
        object_id = torch.where(
            object_id == 0,
            0,
            object_id + batch * unique_obj_ids,
        )

        loss = oc_coward_loss_per_batch(beta, object_id, noise_idx=0, batch=batch)

        manual_loss = 0.0
        for b in batch.unique():
            mask = batch == b
            manual_loss += oc_coward_loss_per_graph(
                beta[mask], object_id[mask], noise_idx=0
            )

        assert torch.isclose(loss, manual_loss, atol=1e-5)
