import torch
from utils.graph import reindex_edge_index


class TestReindexEdgeIndex:
    def test_basic_reindexing(self):
        """Test basic reindexing with simple case."""
        node_ids = torch.tensor([10, 20, 30])
        edge_index = torch.tensor([[10, 20], [20, 30]])
        result = reindex_edge_index(edge_index, node_ids)
        expected = torch.tensor([[0, 1], [1, 2]])
        assert torch.equal(result, expected)

    def test_single_edge(self):
        """Test with single edge."""
        node_ids = torch.tensor([5, 10])
        edge_index = torch.tensor([[5], [10]])
        result = reindex_edge_index(edge_index, node_ids)
        expected = torch.tensor([[0], [1]])
        assert torch.equal(result, expected)

    def test_empty_edge_index(self):
        """Test with empty edge index."""
        node_ids = torch.tensor([1, 2, 3])
        edge_index = torch.tensor([[], []]).long()
        result = reindex_edge_index(edge_index, node_ids)
        assert result.shape == (2, 0)

    def test_self_loops(self):
        """Test with self-loop edges."""
        node_ids = torch.tensor([7, 8, 9])
        edge_index = torch.tensor([[7, 8, 9], [7, 8, 9]])
        result = reindex_edge_index(edge_index, node_ids)
        expected = torch.tensor([[0, 1, 2], [0, 1, 2]])
        assert torch.equal(result, expected)

    def test_non_contiguous_node_ids(self):
        """Test with non-contiguous node IDs."""
        node_ids = torch.tensor([100, 200, 300, 400])
        edge_index = torch.tensor([[100, 200, 300], [200, 300, 100]])
        result = reindex_edge_index(edge_index, node_ids)
        expected = torch.tensor([[0, 1, 2], [1, 2, 0]])
        assert torch.equal(result, expected)

    def test_dtype_is_long(self):
        """Test that output dtype is torch.long."""
        node_ids = torch.tensor([5, 10, 15])
        edge_index = torch.tensor([[5, 10], [10, 15]])
        result = reindex_edge_index(edge_index, node_ids)
        assert result.dtype == torch.long

    def test_multiple_edges_same_nodes(self):
        """Test with multiple edges between same nodes."""
        node_ids = torch.tensor([1, 2])
        edge_index = torch.tensor([[1, 1, 2], [2, 2, 1]])
        result = reindex_edge_index(edge_index, node_ids)
        expected = torch.tensor([[0, 0, 1], [1, 1, 0]])
        assert torch.equal(result, expected)

    def test_large_node_ids(self):
        """Test with large node ID values."""
        node_ids = torch.tensor([1000, 5000, 9999])
        edge_index = torch.tensor([[1000, 5000], [5000, 9999]])
        result = reindex_edge_index(edge_index, node_ids)
        expected = torch.tensor([[0, 1], [1, 2]])
        assert torch.equal(result, expected)

    def test_single_node(self):
        """Test with single node."""
        node_ids = torch.tensor([42])
        edge_index = torch.tensor([[], []]).long()
        result = reindex_edge_index(edge_index, node_ids)
        assert result.shape == (2, 0)

    def test_large_graph(self):
        """Test with larger graph."""
        node_ids = torch.arange(0, 1080)
        edge_index = torch.tensor([[10, 20, 30, 40], [200, 300, 400, 500]])
        result = reindex_edge_index(edge_index, node_ids)
        expected = torch.tensor([[10, 20, 30, 40], [200, 300, 400, 500]])
        assert torch.equal(result, expected)
