import torch
from utils.graph import pack_to_graph_batches
from utils.graph_onnx import pack_to_graph_batches_onnx


def test_pack_to_graph_batches_onnx_basic():
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

    B = int(batch.max().item()) + 1
    L_max = int(
        batch.bincount(minlength=B).max().item()
    )  # max number of nodes per graph

    x_graphs, idx_out, mask_out = pack_to_graph_batches_onnx(x_in, [], batch, B, L_max)
    x_graph = x_graphs[0]  # [B, L_max, D]

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


def test_pack_to_graph_batches_onnx_single_graph():
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

    B = int(batch.max().item()) + 1
    L_max = int(
        batch.bincount(minlength=B).max().item()
    )  # max number of nodes per graph

    x_graphs, idx_out, mask_out = pack_to_graph_batches_onnx(x_in, [], batch, B, L_max)
    x_graph = x_graphs[0]  # [B, L_max, D]

    assert x_graph.shape == (1, 4, 2), f"Expected shape (1, 4, 2), got {x_graph.shape}"
    assert torch.allclose(x_graph[0], x_in), f"Single graph features mismatch"
    assert len(idx_out) == 1, f"Expected 1 index list, got {len(idx_out)}"
    assert torch.equal(idx_out[0], torch.tensor([0, 1, 2, 3])), f"Indices mismatch"
    assert mask_out.shape == (1, 4), f"Expected mask shape (1, 4), got {mask_out.shape}"
    assert torch.all(mask_out[0]), f"All mask values should be True"


def test_pack_to_graph_batches_onnx_multiple_features():
    """Test packing with multiple feature dimensions."""
    batch = torch.tensor([0, 0, 1, 1])  # graph 0: 2 nodes, graph 1: 2 nodes
    x_in = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    # additional feature dimension (e.g. positional features)
    t0_in = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    t1_in = torch.tensor([[0.5, 50], [0.6, 60], [0.7, 70], [0.8, 80]])

    B = int(batch.max().item()) + 1
    L_max = int(
        batch.bincount(minlength=B).max().item()
    )  # max number of nodes per graph

    x_graphs, idx_out, mask_out = pack_to_graph_batches_onnx(
        x_in, [t0_in, t1_in], batch, B, L_max
    )
    x_graph = x_graphs[0]  # [B, L_max, D]
    t0_graph = x_graphs[1]  # [B, L_max, 1]
    t1_graph = x_graphs[2]  # [B, L_max, 2]

    assert x_graph.shape == (2, 2, 2), f"Expected shape (2, 2, 2), got {x_graph.shape}"
    assert t0_graph.shape == (
        2,
        2,
        1,
    ), f"Expected shape (2, 2, 1), got {t0_graph.shape}"
    assert t1_graph.shape == (
        2,
        2,
        2,
    ), f"Expected shape (2, 2, 2), got {t1_graph.shape}"

    # Check graph 0
    assert torch.allclose(
        x_graph[0], torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    ), f"Graph 0 features mismatch"
    assert torch.allclose(
        t0_graph[0], torch.tensor([[0.1], [0.2]])
    ), f"Graph 0 t0 mismatch"
    assert torch.allclose(
        t1_graph[0], torch.tensor([[0.5, 50], [0.6, 60]])
    ), f"Graph 0 t1 mismatch"

    # Check graph 1
    assert torch.allclose(
        x_graph[1], torch.tensor([[3.0, 30.0], [4.0, 40.0]])
    ), f"Graph 1 features mismatch"
    assert torch.allclose(
        t0_graph[1], torch.tensor([[0.3], [0.4]])
    ), f"Graph 1 t0 mismatch"
    assert torch.allclose(
        t1_graph[1], torch.tensor([[0.7, 70], [0.8, 80]])
    ), f"Graph 1 t1 mismatch"


def test_pack_to_graph_batches_onnx_compatibility():
    """Test that pack_to_graph_batches_onnx produces the same output as pack_to_graph_batches."""
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

    # Get output from non-onnx version
    x_graphs_ref, idx_out_ref, mask_out_ref = pack_to_graph_batches(x_in, [], batch)
    x_graph_ref = x_graphs_ref[0]

    # Get output from onnx version
    # Must set B and Lmax in advance and the same as the reference version, since ONNX does not support dynamic shapes or control flow
    B = int(batch.max().item()) + 1
    L_max = int(
        batch.bincount(minlength=B).max().item()
    )  # max number of nodes per graph
    x_graphs_onnx, idx_out_onnx, mask_out_onnx = pack_to_graph_batches_onnx(
        x_in, [], batch, B, L_max
    )
    x_graph_onnx = x_graphs_onnx[0]

    # Check that outputs are the same
    assert torch.allclose(
        x_graph_ref, x_graph_onnx
    ), f"Graph features mismatch between reference and ONNX versions"
    assert len(idx_out_ref) == len(idx_out_onnx), f"Number of index lists mismatch"
    for i in range(len(idx_out_ref)):
        assert torch.equal(
            idx_out_ref[i], idx_out_onnx[i]
        ), f"Index list {i} mismatch between reference and ONNX versions"
    assert torch.equal(
        mask_out_ref, mask_out_onnx
    ), f"Mask mismatch between reference and ONNX versions"
