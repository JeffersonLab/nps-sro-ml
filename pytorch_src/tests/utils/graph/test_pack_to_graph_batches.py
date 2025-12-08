import torch
from utils.graph import pack_to_graph_batches


def test_pack_to_graph_batches_basic():
    """Test basic packing of node features into graph batches."""
    batch = torch.tensor([0, 0, 1, 1, 0, 1])  # 3 nodes in graph 0, 3 nodes in graph 1
    x_in = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ]
    )  # [6, 2]

    x_graph, idx_out, mask_out = pack_to_graph_batches(x_in, batch)

    # Check shape
    assert x_graph.shape == (2, 3, 2), f"Expected shape (2, 3, 2), got {x_graph.shape}"

    # Check that graph 0 contains nodes [0, 1, 4]
    expected_graph_0 = torch.tensor([[1.0, 10.0], [2.0, 20.0], [5.0, 50.0]])
    assert torch.allclose(
        x_graph[0], expected_graph_0
    ), f"Graph 0 mismatch. Got {x_graph[0]}"

    # Check that graph 1 contains nodes [2, 3, 5]
    expected_graph_1 = torch.tensor([[3.0, 30.0], [4.0, 40.0], [6.0, 60.0]])
    assert torch.allclose(
        x_graph[1], expected_graph_1
    ), f"Graph 1 mismatch. Got {x_graph[1]}"

    # Check index lists
    assert len(idx_out) == 2, f"Expected 2 index lists, got {len(idx_out)}"
    assert torch.equal(
        idx_out[0], torch.tensor([0, 1, 4])
    ), f"Graph 0 indices mismatch. Got {idx_out[0]}"
    assert torch.equal(
        idx_out[1], torch.tensor([2, 3, 5])
    ), f"Graph 1 indices mismatch. Got {idx_out[1]}"

    # Check mask
    expected_mask = torch.tensor([[True, True, True], [True, True, True]])
    assert torch.equal(mask_out, expected_mask), f"Mask mismatch. Got {mask_out}"


def test_pack_to_graph_batches_single_graph():
    """Test packing when all nodes belong to a single graph."""
    batch = torch.tensor([0, 0, 0, 0])
    x_in = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    x_graph, idx_out, mask_out = pack_to_graph_batches(x_in, batch)

    assert x_graph.shape == (1, 4, 2), f"Expected shape (1, 4, 2), got {x_graph.shape}"
    assert torch.allclose(x_graph[0], x_in), f"Single graph features mismatch"
    assert len(idx_out) == 1, f"Expected 1 index list, got {len(idx_out)}"
    assert torch.equal(idx_out[0], torch.tensor([0, 1, 2, 3])), f"Indices mismatch"
    assert mask_out.shape == (1, 4), f"Expected mask shape (1, 4), got {mask_out.shape}"
    assert torch.all(mask_out[0]), f"All mask values should be True"


def test_pack_to_graph_batches_unequal_sizes():
    """Test packing when graphs have different numbers of nodes."""
    batch = torch.tensor(
        [0, 0, 1, 1, 1, 2]
    )  # graph 0: 2 nodes, graph 1: 3 nodes, graph 2: 1 node
    x_in = torch.tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
        ]
    )

    x_graph, idx_out, mask_out = pack_to_graph_batches(x_in, batch)

    # Maximum size is 3 nodes
    assert x_graph.shape == (3, 3, 1), f"Expected shape (3, 3, 1), got {x_graph.shape}"

    # Check graph 0 (2 nodes)
    assert torch.allclose(
        x_graph[0, :2], torch.tensor([[1.0], [2.0]])
    ), f"Graph 0 mismatch"

    # Check graph 1 (3 nodes)
    assert torch.allclose(
        x_graph[1], torch.tensor([[3.0], [4.0], [5.0]])
    ), f"Graph 1 mismatch"

    # Check graph 2 (1 node)
    assert torch.allclose(x_graph[2, :1], torch.tensor([[6.0]])), f"Graph 2 mismatch"

    assert len(idx_out) == 3, f"Expected 3 index lists, got {len(idx_out)}"
    assert torch.equal(idx_out[0], torch.tensor([0, 1])), f"Graph 0 indices mismatch"
    assert torch.equal(idx_out[1], torch.tensor([2, 3, 4])), f"Graph 1 indices mismatch"
    assert torch.equal(idx_out[2], torch.tensor([5])), f"Graph 2 indices mismatch"

    assert mask_out.shape == (3, 3), f"Expected mask shape (3, 3), got {mask_out.shape}"
    assert torch.equal(
        mask_out,
        torch.tensor([[True, True, False], [True, True, True], [True, False, False]]),
    ), f"Mask mismatch"
