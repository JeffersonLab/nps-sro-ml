import torch
from utils.graph import find_connected_components_undirected_dfs


def test_find_connected_components_undirected_dfs_simple():
    edge_index = torch.tensor([[0, 2, 3, 5, 7, 9], [1, 3, 4, 6, 8, 9]])
    num_nodes = 10

    components = find_connected_components_undirected_dfs(num_nodes, edge_index)

    expected_components = [
        torch.tensor([0, 1]),
        torch.tensor([2, 3, 4]),
        torch.tensor([5, 6]),
        torch.tensor([7, 8]),
        torch.tensor([9]),
    ]

    assert len(components) == len(
        expected_components
    ), f"Expected {len(expected_components)} components, got {len(components)}"

    for comp, expected in zip(components, expected_components):
        assert torch.equal(
            torch.sort(comp).values, torch.sort(expected).values
        ), f"Component mismatch. Got {comp}, expected {expected}"


def test_find_connected_components_undirected_dfs_multiple_components():
    """Test finding multiple disconnected components."""
    # Graph: 0-1, 2-3-4, 5-6, 7, 8-9
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 3, 4, 5, 6, 8, 9], [1, 0, 3, 2, 4, 3, 6, 5, 9, 8]]
    )
    num_nodes = 10

    components = find_connected_components_undirected_dfs(num_nodes, edge_index)

    # Should have 4 components: {0,1}, {2,3,4}, {5,6}, {7}, {8,9}
    assert len(components) == 4, f"Expected 4 components, got {len(components)}"

    # Sort each component for comparison
    sorted_components = [sorted(comp.tolist()) for comp in components]
    sorted_components.sort()

    expected = [[0, 1], [2, 3, 4], [5, 6], [8, 9]]
    assert (
        sorted_components == expected
    ), f"Expected {expected}, got {sorted_components}"


def test_find_connected_components_undirected_dfs_fully_connected():
    """Test a fully connected graph."""
    # Complete graph with 4 nodes
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]])
    num_nodes = 4

    components = find_connected_components_undirected_dfs(num_nodes, edge_index)

    assert len(components) == 1, f"Expected 1 component, got {len(components)}"
    assert sorted(components[0].tolist()) == [
        0,
        1,
        2,
        3,
    ], "All nodes should be in one component"


def test_find_connected_components_undirected_dfs_no_edges():
    """Test graph with no edges (all isolated nodes)."""
    edge_index = torch.tensor([[], []], dtype=torch.long)
    num_nodes = 5

    components = find_connected_components_undirected_dfs(num_nodes, edge_index)

    assert (
        len(components) == 0
    ), f"Expected 0 components (no edges to traverse), got {len(components)}"


def test_find_connected_components_undirected_dfs_linear_chain():
    """Test a linear chain graph."""
    # 0-1-2-3-4
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    num_nodes = 5

    components = find_connected_components_undirected_dfs(num_nodes, edge_index)

    assert len(components) == 1, f"Expected 1 component, got {len(components)}"
    assert sorted(components[0].tolist()) == [
        0,
        1,
        2,
        3,
        4,
    ], "All nodes should be connected"


def test_find_connected_components_undirected_dfs_self_loops():
    """Test graph with self-loops."""
    # 0-0, 1-2, 3-3
    edge_index = torch.tensor([[0, 0, 1, 2, 3, 3], [0, 0, 2, 1, 3, 3]])
    num_nodes = 4

    components = find_connected_components_undirected_dfs(num_nodes, edge_index)

    sorted_components = [sorted(comp.tolist()) for comp in components]
    sorted_components.sort()

    expected = [[0], [1, 2], [3]]
    assert (
        sorted_components == expected
    ), f"Expected {expected}, got {sorted_components}"


def test_find_connected_components_undirected_dfs_star_graph():
    """Test a star graph (one center node connected to all others)."""
    # Center node 0 connected to 1, 2, 3, 4
    edge_index = torch.tensor([[0, 1, 0, 2, 0, 3, 0, 4], [1, 0, 2, 0, 3, 0, 4, 0]])
    num_nodes = 5

    components = find_connected_components_undirected_dfs(num_nodes, edge_index)

    assert len(components) == 1, f"Expected 1 component, got {len(components)}"
    assert sorted(components[0].tolist()) == [
        0,
        1,
        2,
        3,
        4,
    ], "All nodes should be connected"


def test_find_connected_components_undirected_dfs_cycle():
    """Test a cycle graph."""
    # 0-1-2-3-0 (cycle)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
    num_nodes = 4

    components = find_connected_components_undirected_dfs(num_nodes, edge_index)

    assert len(components) == 1, f"Expected 1 component, got {len(components)}"
    assert sorted(components[0].tolist()) == [
        0,
        1,
        2,
        3,
    ], "All nodes in cycle should be connected"


def test_find_connected_components_undirected_dfs_isolated():
    edge_index = torch.tensor(
        [[0, 1, 2, 4], [1, 2, 3, 5]]
    )  # chain 0-1-2-3, pair 4-5, nodes 6–7 isolated
    num_nodes = 8

    components = find_connected_components_undirected_dfs(num_nodes, edge_index)
    expected = [
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([4, 5]),
        # 6,7 are singletons but SHOULD NOT COUNT per your rules
    ]

    assert len(components) == len(
        expected
    ), f"Expected {len(expected)} components, got {len(components)}"


def test_dst_only_node_fails_old_dfs():
    edge_index = torch.tensor(
        [[0], [1]]
    )  # Edge: 0 -- 1   BUT node 1 appears only in dst
    num_nodes = 2

    components = find_connected_components_undirected_dfs(num_nodes, edge_index)

    # Expected: one component [0,1]
    assert len(components) == 1, f"Expected 1 component, got {len(components)}"
    assert torch.equal(
        torch.sort(components[0]).values, torch.tensor([0, 1])
    ), f"Component mismatch: got {components}"


def test_disconnected_by_src_order():
    edge_index = torch.tensor([[2, 2, 2, 2], [3, 4, 5, 6]])
    num_nodes = 7

    components = find_connected_components_undirected_dfs(num_nodes, edge_index)

    # Expected: single component [2,3,4,5,6]
    expected = [2, 3, 4, 5, 6]

    assert len(components) == 1, f"Expected 1 component, got {len(components)}"
    assert sorted(components[0].tolist()) == expected, f"Got {components}"


def test_dst_only_back_edge_breaks():
    edge_index = torch.tensor([[0, 1], [2, 0]])
    num_nodes = 3

    # Undirected edges:
    # 0 -- 2
    # 1 -- 0
    # So all nodes should be connected
    components = find_connected_components_undirected_dfs(num_nodes, edge_index)

    expected = [0, 1, 2]

    assert len(components) == 1, f"Expected 1 component, got {len(components)}"
    assert sorted(components[0].tolist()) == expected, f"Got {components}"
