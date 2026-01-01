import pytest
import torch
from utils.graph import edge_to_adj_matrix


def test_edge_to_adj_matrix_basic():
    """Test basic conversion from edge index to adjacency matrix."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    num_nodes = 3

    adj = edge_to_adj_matrix(edge_index, num_nodes)

    assert adj.shape == (3, 3)
    assert adj.dtype == torch.bool
    assert adj[0, 1] == True
    assert adj[1, 2] == True
    assert adj[2, 0] == True
    assert adj[0, 0] == False
    assert adj[1, 1] == False
    assert adj[2, 2] == False


def test_edge_to_adj_matrix_directed():
    """Test that adjacency matrix respects edge directionality."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    num_nodes = 2

    adj = edge_to_adj_matrix(edge_index, num_nodes)

    assert adj[0, 1] == True
    assert adj[1, 0] == True


def test_edge_to_adj_matrix_self_loops():
    """Test handling of self-loops."""
    edge_index = torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long)
    num_nodes = 3

    adj = edge_to_adj_matrix(edge_index, num_nodes)

    assert adj[0, 0] == True  # self-loop
    assert adj[1, 1] == True  # self-loop
    assert adj[1, 2] == True


def test_edge_to_adj_matrix_empty():
    """Test with no edges."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    num_nodes = 3

    adj = edge_to_adj_matrix(edge_index, num_nodes)

    assert adj.shape == (3, 3)
    assert not adj.any()


def test_edge_to_adj_matrix_single_node():
    """Test with a single node and no edges."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    num_nodes = 1

    adj = edge_to_adj_matrix(edge_index, num_nodes)

    assert adj.shape == (1, 1)
    assert adj[0, 0] == False


def test_edge_to_adj_matrix_complete_graph():
    """Test with a complete graph (all nodes connected)."""
    num_nodes = 4
    # Create edges for complete graph (excluding self-loops)
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    adj = edge_to_adj_matrix(edge_index, num_nodes)

    assert adj.shape == (num_nodes, num_nodes)
    # All off-diagonal elements should be True
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                assert adj[i, j] == True
            else:
                assert adj[i, j] == False


def test_edge_to_adj_matrix_device_cpu():
    """Test that output is on the same device as input (CPU)."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    num_nodes = 3

    adj = edge_to_adj_matrix(edge_index, num_nodes)

    assert adj.device == edge_index.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_edge_to_adj_matrix_device_cuda():
    """Test that output is on the same device as input (CUDA)."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device='cuda')
    num_nodes = 3

    adj = edge_to_adj_matrix(edge_index, num_nodes)

    assert adj.device == edge_index.device
    assert adj.is_cuda


def test_edge_to_adj_matrix_duplicate_edges():
    """Test with duplicate edges (should still be True in adjacency matrix)."""
    edge_index = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
    num_nodes = 3

    adj = edge_to_adj_matrix(edge_index, num_nodes)

    assert adj[0, 1] == True
    assert adj[1, 2] == True


def test_edge_to_adj_matrix_isolated_nodes():
    """Test with isolated nodes (nodes with no edges)."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    num_nodes = 4  # Node 3 is isolated

    adj = edge_to_adj_matrix(edge_index, num_nodes)

    assert adj.shape == (4, 4)
    assert adj[3].sum() == 0  # Row for isolated node should be all False
    assert adj[:, 3].sum() == 0  # Column for isolated node should be all False


def test_edge_to_adj_matrix_large_graph():
    """Test with a larger graph to check performance and correctness."""
    num_nodes = 1000
    edges = []
    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    adj = edge_to_adj_matrix(edge_index, num_nodes)

    assert adj.shape == (num_nodes, num_nodes)
    for i in range(num_nodes - 1):
        assert adj[i, i + 1] == True
    assert adj.sum() == (num_nodes - 1)


def test_edge_to_adj_matrix_invalid_shape():
    """Test with invalid edge_index shape."""
    edge_index = torch.tensor([0, 1, 2], dtype=torch.long)  # 1D instead of 2D
    num_nodes = 3

    with pytest.raises(ValueError, match="edge_index must have shape"):
        edge_to_adj_matrix(edge_index, num_nodes)


def test_edge_to_adj_matrix_wrong_first_dim():
    """Test with wrong first dimension size."""
    edge_index = torch.tensor([[0, 1, 2]], dtype=torch.long)  # shape [1, 3]
    num_nodes = 3

    with pytest.raises(ValueError, match="edge_index must have shape"):
        edge_to_adj_matrix(edge_index, num_nodes)


def test_edge_to_adj_matrix_out_of_range_nodes():
    """Test with node indices out of range."""
    edge_index = torch.tensor(
        [[0, 1], [1, 3]], dtype=torch.long
    )  # node 3 doesn't exist
    num_nodes = 3

    with pytest.raises(ValueError, match="out-of-range"):
        edge_to_adj_matrix(edge_index, num_nodes)


def test_edge_to_adj_matrix_negative_indices():
    """Test with negative node indices."""
    edge_index = torch.tensor([[0, -1], [1, 2]], dtype=torch.long)
    num_nodes = 3

    with pytest.raises(ValueError, match="out-of-range"):
        edge_to_adj_matrix(edge_index, num_nodes)
