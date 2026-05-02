import torch
import pytest
from models.oc_inference import (
    MultiPulseOCInferencer,
    PulseOCInferencer,
    WaveformOCInferencer,
    oc_inference_per_graph,
    oc_inference_per_batch,
)
from models.oc_base import ObjectCondensationBaseModel


class DummyModel(ObjectCondensationBaseModel):
    def __init__(self, input_type="pulse_set"):
        super().__init__(input_type=input_type)
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, pos, fea_mask, node_mask):
        return x[..., 0, :] if x.dim() == 4 else x, node_mask.unsqueeze(-1).to(x.dtype)


class DummyData:
    def __init__(self, x, pos, y, batch=None):
        self.x = x
        self.pos = pos
        self.y = y
        if batch is not None:
            self.batch = batch

    def to(self, device):
        self.x = self.x.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        if hasattr(self, "batch"):
            self.batch = self.batch.to(device)
        return self


class DummyMultiPulseModel(ObjectCondensationBaseModel):
    def __init__(self, input_type="pulse_set", num_pulse_tokens=2):
        super().__init__(input_type=input_type)
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.num_pulse_tokens = num_pulse_tokens

    def forward(self, x, pos, fea_mask, node_mask):
        batch_size, num_nodes = node_mask.shape
        pulse_beta = torch.tensor(
            [[[0.9, 0.2], [0.1, 0.8]]],
            dtype=x.dtype,
            device=x.device,
        ).expand(batch_size, num_nodes, -1)
        pulse_score = torch.tensor(
            [[[0.95, 0.1], [0.2, 0.9]]],
            dtype=x.dtype,
            device=x.device,
        ).expand(batch_size, num_nodes, -1)
        pulse_x_c = torch.tensor(
            [[[[0.0, 0.0], [10.0, 10.0]], [[2.0, 0.0], [2.1, 0.0]]]],
            dtype=x.dtype,
            device=x.device,
        ).expand(batch_size, num_nodes, -1, -1)
        token_mask = node_mask.unsqueeze(-1).expand(batch_size, num_nodes, self.num_pulse_tokens)

        return {
            "pulse_beta": pulse_beta * token_mask.to(pulse_beta.dtype),
            "pulse_x_c": pulse_x_c * token_mask.unsqueeze(-1).to(pulse_x_c.dtype),
            "pulse_score": pulse_score * token_mask.to(pulse_score.dtype),
            "pulse_time": torch.zeros_like(pulse_score),
            "pulse_charge": torch.zeros_like(pulse_score),
            "proposal_score": pulse_score * token_mask.to(pulse_score.dtype),
            "proposal_time": torch.zeros_like(pulse_score),
            "proposal_width": torch.zeros_like(pulse_score),
            "proposal_amplitude": torch.zeros_like(pulse_score),
            "token_mask": token_mask,
        }


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


def test_pulse_inferencer_prepares_packed_inputs_like_trainer():
    model = DummyModel(input_type="pulse_set")
    inferencer = PulseOCInferencer(model)
    data = DummyData(
        x=torch.tensor(
            [
                [1.0, 10.0, 2.0, 20.0],
                [3.0, 30.0, 0.0, 0.0],
                [4.0, 40.0, 5.0, 50.0],
            ],
            dtype=torch.float32,
        ),
        pos=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        ),
        y=torch.tensor([[1], [2], [3]], dtype=torch.float32),
        batch=torch.tensor([0, 0, 1], dtype=torch.long),
    )

    x_graph, pos_graph, fea_mask, node_mask, _ = model.prepare_graph_inputs(
        data.x,
        data.pos,
        data.batch,
    )

    assert x_graph.shape == (2, 2, 2, 2)
    assert pos_graph.shape == (2, 2, 2)
    assert torch.equal(node_mask, torch.tensor([[True, True], [True, False]]))
    assert torch.equal(
        fea_mask,
        torch.tensor(
            [
                [[True, True], [True, False]],
                [[True, True], [False, False]],
            ]
        ),
    )


def test_waveform_inferencer_preprocesses_waveforms_like_trainer(tmp_path):
    cfg_path = tmp_path / "vme.csv"
    cfg_path.write_text("channel,FADC250_ALLCH_PED\n0,1.0\n1,2.0\n")

    model = DummyModel(input_type="waveform")
    inferencer = WaveformOCInferencer(
        model,
        config={"trainer": {"args": {"vme_config": str(cfg_path)}}},
    )
    data = DummyData(
        x=torch.tensor([[10.0, 11.0], [20.0, 21.0]], dtype=torch.float32),
        pos=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        y=torch.tensor([[1], [2]], dtype=torch.float32),
    )

    processed = model.preprocess_features(data)
    batch = model.get_batch_vector(processed, getattr(data, "batch", None))
    x_graph, _, fea_mask, node_mask, _ = model.prepare_graph_inputs(
        processed,
        data.pos,
        batch,
    )

    assert torch.equal(
        x_graph,
        torch.tensor([[[9.0, 10.0], [18.0, 19.0]]], dtype=torch.float32),
    )
    assert torch.equal(node_mask, torch.tensor([[True, True]]))
    assert torch.equal(fea_mask, torch.tensor([[[True, True], [True, True]]]))


def test_multi_pulse_inferencer_predicts_per_slot_clusters():
    model = DummyMultiPulseModel(input_type="pulse_set", num_pulse_tokens=2)
    inferencer = MultiPulseOCInferencer(model)
    data = DummyData(
        x=torch.tensor(
            [
                [1.0, 10.0, 0.0, 0.0],
                [2.0, 20.0, 3.0, 30.0],
            ],
            dtype=torch.float32,
        ),
        pos=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0]],
            dtype=torch.float32,
        ),
        y=torch.tensor([[11, -1], [-1, 22]], dtype=torch.long),
    )

    results = inferencer.infer_dataloader([data]).to_dataframe()

    assert "pulse_cluster_ids_0" in results.columns
    assert "pulse_cluster_ids_1" in results.columns
    assert "pulse_object_ids_0" in results.columns
    assert "pulse_object_ids_1" in results.columns
    assert results["pulse_cluster_ids_0"].tolist() == [0, -1]
    assert results["pulse_cluster_ids_1"].tolist() == [-1, 1]
    assert results["pulse_object_ids_0"].tolist()[0] != -1
    assert results["pulse_object_ids_1"].tolist()[0] == -1
    assert results["pulse_object_ids_0"].tolist()[1] == -1
    assert results["pulse_object_ids_1"].tolist()[1] != -1
