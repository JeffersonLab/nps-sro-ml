import torch
from utils.graph import find_local_edge_index


def test_find_local_edge_index_single_graph():
    """Test finding local edge index for a single graph in batch."""
    batch = torch.tensor([0, 0, 0, 0])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    edge_index_b = find_local_edge_index(edge_index, batch, b=0)

    expected = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert edge_index_b.shape == expected.shape
    assert torch.equal(edge_index_b, expected)


def test_find_local_edge_index_multiple_graphs():
    """Test finding local edge index for specific graph in multi-graph batch."""
    # Graph 0: nodes 0,1,2; Graph 1: nodes 3,4,5
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])

    # Get edges for graph 0
    edge_index_0 = find_local_edge_index(edge_index, batch, b=0)
    expected_0 = torch.tensor([[0, 1], [1, 2]])
    assert torch.equal(edge_index_0, expected_0)

    # Get edges for graph 1
    edge_index_1 = find_local_edge_index(edge_index, batch, b=1)
    expected_1 = torch.tensor([[0, 1], [1, 2]])
    assert torch.equal(edge_index_1, expected_1)


def test_find_local_edge_index_no_edges():
    """Test finding local edge index when graph has no edges."""
    batch = torch.tensor([0, 0, 1, 1])
    edge_index = torch.tensor([[0], [1]])  # Only graph 0 has edges

    edge_index_1 = find_local_edge_index(edge_index, batch, b=1)

    assert edge_index_1.shape == (2, 0)
    assert edge_index_1.numel() == 0


def test_find_local_edge_index_nonexistent_graph():
    """Test finding local edge index for non-existent graph."""
    batch = torch.tensor([0, 0, 1, 1])
    edge_index = torch.tensor([[0, 1], [1, 2]])

    edge_index_2 = find_local_edge_index(edge_index, batch, b=2)

    assert edge_index_2.shape == (2, 0)
    assert edge_index_2.numel() == 0


def test_find_local_edge_index_non_sequential_nodes():
    """Test with non-sequential node indices in batch."""
    # Graph 0: nodes 0,2,4; Graph 1: nodes 1,3,5
    batch = torch.tensor([0, 1, 0, 1, 0, 1])
    edge_index = torch.tensor([[0, 2, 4, 1, 3], [2, 4, 0, 3, 5]])

    edge_index_0 = find_local_edge_index(edge_index, batch, b=0)
    expected_0 = torch.tensor([[0, 1, 2], [1, 2, 0]])
    assert torch.equal(edge_index_0, expected_0)

    edge_index_1 = find_local_edge_index(edge_index, batch, b=1)
    expected_1 = torch.tensor([[0, 1], [1, 2]])
    assert torch.equal(edge_index_1, expected_1)


def test_find_local_edge_index_self_loops():
    """Test finding local edge index with self-loops."""
    batch = torch.tensor([0, 0, 1, 1])
    edge_index = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 1, 3, 3]])

    edge_index_0 = find_local_edge_index(edge_index, batch, b=0)
    expected_0 = torch.tensor([[0, 0, 1], [0, 1, 1]])
    assert torch.equal(edge_index_0, expected_0)

    edge_index_1 = find_local_edge_index(edge_index, batch, b=1)
    expected_1 = torch.tensor([[0, 1], [1, 1]])
    assert torch.equal(edge_index_1, expected_1)


def test_find_local_edge_index_fully_connected():
    """Test finding local edge index for fully connected graph."""
    batch = torch.tensor([0, 0, 0, 1, 1])
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 4], [1, 2, 0, 2, 0, 1, 4, 3]])

    edge_index_0 = find_local_edge_index(edge_index, batch, b=0)
    expected_0 = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    assert torch.equal(edge_index_0, expected_0)


def test_find_local_edge_index_empty_edge_index():
    """Test finding local edge index with empty edge index."""
    batch = torch.tensor([0, 0, 1, 1])
    edge_index = torch.tensor([[], []], dtype=torch.long)

    edge_index_0 = find_local_edge_index(edge_index, batch, b=0)

    assert edge_index_0.shape == (2, 0)
    assert edge_index_0.numel() == 0
