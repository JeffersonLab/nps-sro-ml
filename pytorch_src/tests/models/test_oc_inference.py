import torch
import pytest
from models.oc_inference import oc_inference_per_graph, oc_inference_per_batch


def test_oc_inference_per_graph_basic():
    """Test basic clustering functionality."""
    x = torch.tensor([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    beta = torch.tensor([0.8, 0.3, 0.9, 0.2])

    cluster_ids, min_d = oc_inference_per_graph(x, beta, beta_thres=0.4, dist_thres=1.0)

    assert cluster_ids.shape == (4,)
    assert min_d.shape == (4,)
    assert cluster_ids.dtype == torch.long
    assert cluster_ids[0] == cluster_ids[1]  # Close points should cluster together
    assert cluster_ids[2] != cluster_ids[0]  # Far points should be different clusters


def test_oc_inference_per_graph_no_seeds():
    """Test when no points exceed beta threshold."""
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    beta = torch.tensor([0.1, 0.2, 0.3])

    cluster_ids, min_d = oc_inference_per_graph(x, beta, beta_thres=0.4, dist_thres=0.8)

    assert torch.all(cluster_ids == 0)  # All background
    assert torch.all(min_d == float('inf'))


def test_oc_inference_per_graph_all_background_by_distance():
    """Test when seeds exist but all points are too far."""
    x = torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    beta = torch.tensor([0.9, 0.1, 0.1])

    cluster_ids, min_d = oc_inference_per_graph(x, beta, beta_thres=0.4, dist_thres=1.0)

    assert cluster_ids[0] != 0  # Seed point gets cluster ID
    assert cluster_ids[1] == 0  # Too far from seed
    assert cluster_ids[2] == 0  # Too far from seed


def test_oc_inference_per_graph_multiple_clusters():
    """Test formation of multiple distinct clusters."""
    x = torch.tensor([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    beta = torch.tensor([0.9, 0.2, 0.8, 0.3])

    cluster_ids, min_d = oc_inference_per_graph(x, beta, beta_thres=0.5, dist_thres=1.0)

    assert cluster_ids[0] != cluster_ids[2]  # Two separate clusters
    assert cluster_ids[0] == cluster_ids[1]  # Nearby points cluster with seed
    assert cluster_ids[2] == cluster_ids[3]  # Nearby points cluster with seed


def test_oc_inference_per_graph_beta_2d():
    """Test with beta as 2D tensor [num_nodes, 1]."""
    x = torch.tensor([[0.0, 0.0], [0.1, 0.1]])
    beta = torch.tensor([[0.8], [0.3]])

    cluster_ids, min_d = oc_inference_per_graph(x, beta, beta_thres=0.4, dist_thres=1.0)

    assert cluster_ids.shape == (2,)
    assert min_d.shape == (2,)


def test_oc_inference_per_graph_custom_bkg_idx():
    """Test with custom background index."""
    x = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    beta = torch.tensor([0.9, 0.1])

    cluster_ids, min_d = oc_inference_per_graph(
        x, beta, beta_thres=0.4, dist_thres=1.0, bkg_idx=99
    )

    assert cluster_ids[0] == 100  # bkg_idx + 1 for first cluster
    assert cluster_ids[1] == 99  # Background uses bkg_idx


def test_oc_inference_per_graph_single_point():
    """Test with single point."""
    x = torch.tensor([[1.0, 1.0]])
    beta = torch.tensor([0.9])

    cluster_ids, min_d = oc_inference_per_graph(x, beta, beta_thres=0.4, dist_thres=1.0)

    assert cluster_ids.shape == (1,)
    assert cluster_ids[0] == 1  # Gets cluster ID 1


def test_oc_inference_per_graph_distance_threshold():
    """Test that distance threshold is properly applied."""
    x = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.5, 0.0]])
    beta = torch.tensor([0.9, 0.1, 0.1])

    cluster_ids, min_d = oc_inference_per_graph(x, beta, beta_thres=0.4, dist_thres=1.0)

    assert cluster_ids[0] == 1  # Seed
    assert cluster_ids[1] == 1  # Within threshold
    assert cluster_ids[2] == 0  # Beyond threshold
    assert min_d[0] == 0.0  # Distance to itself
    assert min_d[1] == pytest.approx(0.5)
    assert min_d[2] == pytest.approx(1.5)

    def test_oc_inference_per_batch_basic():
        """Test basic clustering functionality with batch."""
        x = torch.tensor([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
        beta = torch.tensor([0.8, 0.3, 0.9, 0.2])
        batch = torch.tensor([0, 0, 0, 0])

        cluster_ids, min_d = oc_inference_per_batch(
            x, beta, batch, beta_thres=0.4, dist_thres=1.0
        )

        assert cluster_ids.shape == (4,)
        assert min_d.shape == (4,)
        assert cluster_ids.dtype == torch.long
        assert cluster_ids[0] == cluster_ids[1]  # Close points should cluster together
        assert (
            cluster_ids[2] != cluster_ids[0]
        )  # Far points should be different clusters


def test_oc_inference_per_batch_multiple_graphs():
    """Test clustering with multiple graphs in batch."""
    x = torch.tensor([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    beta = torch.tensor([0.8, 0.3, 0.9, 0.2])
    batch = torch.tensor([0, 0, 1, 1])

    cluster_ids, min_d = oc_inference_per_batch(
        x, beta, batch, beta_thres=0.4, dist_thres=1.0
    )

    assert cluster_ids.shape == (4,)
    # Points from different graphs should not cluster together even if close in space
    assert cluster_ids[0] != cluster_ids[2]
    assert cluster_ids[0] == cluster_ids[1]  # Same graph, close points
    assert cluster_ids[2] == cluster_ids[3]  # Same graph, close points


def test_oc_inference_per_batch_no_seeds():
    """Test when no points exceed beta threshold."""
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    beta = torch.tensor([0.1, 0.2, 0.3])
    batch = torch.tensor([0, 0, 1])

    cluster_ids, min_d = oc_inference_per_batch(
        x, beta, batch, beta_thres=0.4, dist_thres=0.8
    )

    assert torch.all(cluster_ids == 0)  # All background
    assert torch.all(min_d == float('inf'))


def test_oc_inference_per_batch_seed_isolation():
    """Test that seeds only affect their own graph."""
    x = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.0, 0.0], [0.1, 0.1]])
    beta = torch.tensor([0.9, 0.2, 0.1, 0.2])  # Only first point is seed
    batch = torch.tensor([0, 0, 1, 1])

    cluster_ids, min_d = oc_inference_per_batch(
        x, beta, batch, beta_thres=0.4, dist_thres=1.0
    )

    assert cluster_ids[0] == 1  # Seed in graph 0
    assert cluster_ids[1] == 1  # Clustered with seed in graph 0
    assert cluster_ids[2] == 0  # No seed in graph 1, background
    assert cluster_ids[3] == 0  # No seed in graph 1, background


def test_oc_inference_per_batch_distance_threshold():
    """Test that distance threshold is properly applied per graph."""
    x = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.5, 0.0], [0.0, 0.0], [0.5, 0.0]])
    beta = torch.tensor([0.9, 0.1, 0.1, 0.9, 0.1])
    batch = torch.tensor([0, 0, 0, 1, 1])

    cluster_ids, min_d = oc_inference_per_batch(
        x, beta, batch, beta_thres=0.4, dist_thres=1.0
    )

    # Graph 0
    assert cluster_ids[0] == 1  # Seed
    assert cluster_ids[1] == 1  # Within threshold
    assert cluster_ids[2] == 0  # Beyond threshold

    # Graph 1
    assert cluster_ids[3] == 2  # Seed (new cluster ID)
    assert cluster_ids[4] == 2  # Within threshold


def test_oc_inference_per_batch_beta_2d():
    """Test with beta as 2D tensor [num_nodes, 1]."""
    x = torch.tensor([[0.0, 0.0], [0.1, 0.1]])
    beta = torch.tensor([[0.8], [0.3]])
    batch = torch.tensor([0, 0])

    cluster_ids, min_d = oc_inference_per_batch(
        x, beta, batch, beta_thres=0.4, dist_thres=1.0
    )

    assert cluster_ids.shape == (2,)
    assert min_d.shape == (2,)


def test_oc_inference_per_batch_custom_bkg_idx():
    """Test with custom background index."""
    x = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    beta = torch.tensor([0.9, 0.1])
    batch = torch.tensor([0, 0])

    cluster_ids, min_d = oc_inference_per_batch(
        x, beta, batch, beta_thres=0.4, dist_thres=1.0, bkg_idx=99
    )

    assert cluster_ids[0] == 100  # bkg_idx + 1 for first cluster
    assert cluster_ids[1] == 99  # Background uses bkg_idx


def test_oc_inference_per_batch_single_node_per_graph():
    """Test with single node in each graph."""
    x = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    beta = torch.tensor([0.9, 0.8])
    batch = torch.tensor([0, 1])

    cluster_ids, min_d = oc_inference_per_batch(
        x, beta, batch, beta_thres=0.4, dist_thres=1.0
    )

    assert cluster_ids[0] == 1  # First cluster
    assert cluster_ids[1] == 2  # Second cluster (different graph)
    assert min_d[0] == 0.0
    assert min_d[1] == 0.0


def test_oc_inference_per_batch_cross_graph_no_clustering():
    """Test that spatially close points from different graphs don't cluster."""
    x = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # Same coordinates
    beta = torch.tensor([0.9, 0.9])  # Both are seeds
    batch = torch.tensor([0, 1])  # Different graphs

    cluster_ids, min_d = oc_inference_per_batch(
        x, beta, batch, beta_thres=0.4, dist_thres=1.0
    )

    assert cluster_ids[0] != cluster_ids[1]  # Must have different cluster IDs
    assert cluster_ids[0] == 1
    assert cluster_ids[1] == 2


def test_oc_inference_per_batch_empty_graph():
    """Test behavior with graph containing only non-seeds."""
    x = torch.tensor([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    beta = torch.tensor([0.9, 0.2, 0.1, 0.2])  # Only first point is seed
    batch = torch.tensor([0, 0, 1, 1])

    cluster_ids, min_d = oc_inference_per_batch(
        x, beta, batch, beta_thres=0.4, dist_thres=1.0
    )

    # Graph 0 has seed
    assert cluster_ids[0] == 1
    assert cluster_ids[1] == 1

    # Graph 1 has no seeds
    assert cluster_ids[2] == 0
    assert cluster_ids[3] == 0
    assert min_d[2] == float('inf')
    assert min_d[3] == float('inf')
