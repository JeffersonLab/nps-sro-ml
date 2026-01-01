import torch
from utils.graph import pack_to_graph_batches, reorder_from_graph_batches


def test_reorder_from_graph_batches_simple():
    """Test reordering with a simple two-graph case."""
    x_graph = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]]
    )  # [2, 2, 2]
    idx_out = [torch.tensor([0, 2]), torch.tensor([1, 3])]

    x_reordered = reorder_from_graph_batches(x_graph, idx_out)

    expected = torch.tensor([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0], [4.0, 40.0]])
    assert torch.allclose(
        x_reordered, expected
    ), f"Expected {expected}, got {x_reordered}"


def test_reorder_from_graph_batches_unequal_sizes():
    """Test reordering when graphs have different numbers of nodes."""
    # Graph 0: 2 nodes, Graph 1: 3 nodes, Graph 2: 1 node
    x_graph = torch.tensor(
        [[[1.0], [2.0], [0.0]], [[3.0], [4.0], [5.0]], [[6.0], [0.0], [0.0]]]
    )  # [3, 3, 1]
    idx_out = [torch.tensor([0, 1]), torch.tensor([2, 3, 4]), torch.tensor([5])]

    x_reordered = reorder_from_graph_batches(x_graph, idx_out)

    expected = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    assert x_reordered.shape == (
        6,
        1,
    ), f"Expected shape (6, 1), got {x_reordered.shape}"
    assert torch.allclose(
        x_reordered, expected
    ), f"Expected {expected}, got {x_reordered}"


def test_reorder_from_graph_batches_single_graph():
    """Test reordering with a single graph."""
    x_graph = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # [1, 3, 2]
    idx_out = [torch.tensor([0, 1, 2])]

    x_reordered = reorder_from_graph_batches(x_graph, idx_out)

    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert torch.allclose(
        x_reordered, expected
    ), f"Expected {expected}, got {x_reordered}"


def test_reorder_from_graph_batches_non_sequential_indices():
    """Test reordering with non-sequential global indices."""
    x_graph = torch.tensor(
        [[[10.0], [20.0], [30.0]], [[40.0], [50.0], [0.0]]]
    )  # [2, 3, 1]
    idx_out = [torch.tensor([5, 2, 8]), torch.tensor([0, 3])]

    x_reordered = reorder_from_graph_batches(x_graph, idx_out)

    # Original order should be: idx 0, 2, 3, 5, 8
    expected = torch.tensor([[40.0], [20.0], [50.0], [10.0], [30.0]])
    assert x_reordered.shape == (
        5,
        1,
    ), f"Expected shape (5, 1), got {x_reordered.shape}"
    assert torch.allclose(
        x_reordered, expected
    ), f"Expected {expected}, got {x_reordered}"


def test_reorder_from_graph_batches_roundtrip():
    """Test that pack_to_graph_batches followed by reorder_from_graph_batches recovers original."""
    batch = torch.tensor([1, 0, 1, 0, 1])
    x_original = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    )

    x_graph, idx_out, mask_out = pack_to_graph_batches(x_original, batch)
    x_recovered = reorder_from_graph_batches(x_graph, idx_out)

    assert torch.allclose(
        x_recovered, x_original
    ), f"Roundtrip failed. Expected {x_original}, got {x_recovered}"
    assert (
        mask_out.shape[0] == x_graph.shape[0]
    ), f"Batch size mismatch between mask and x_graph"
