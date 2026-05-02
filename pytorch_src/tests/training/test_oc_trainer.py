import torch
from torch import nn

from models.oc_base import ObjectCondensationBaseModel
from training.oc_multi_pulse_trainer import MultiPulseOCTrainer
from training.oc_trainer import (
    WaveformOCTrainer,
    create_sample_mask,
)


class DummyModel(ObjectCondensationBaseModel):
    def __init__(self, input_type="waveform"):
        super().__init__(input_type=input_type)
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, pos, fea_mask, node_mask):
        return x, x[..., :1]


class DummyData:
    def __init__(self, x, pos, y, batch=None, edge_index=None):
        self.x = x
        self.pos = pos
        self.y = y
        if batch is not None:
            self.batch = batch
        if edge_index is not None:
            self.edge_index = edge_index

    def to(self, device):
        self.x = self.x.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        if hasattr(self, "batch"):
            self.batch = self.batch.to(device)
        if hasattr(self, "edge_index"):
            self.edge_index = self.edge_index.to(device)
        return self


class DummyMultiPulseModel(ObjectCondensationBaseModel):
    def __init__(self, input_type="waveform", num_pulse_tokens=2):
        super().__init__(input_type=input_type)
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.num_pulse_tokens = num_pulse_tokens

    def forward(self, x, pos, fea_mask, node_mask):
        proposal = self.propose_pulses(x, pos, fea_mask, node_mask)
        cluster_outputs = self.cluster_pulses(proposal)
        gate = cluster_outputs["cluster_token_mask"].to(
            cluster_outputs["cluster_seedness_beta"].dtype
        )
        node_gate = node_mask.unsqueeze(-1).to(gate.dtype)
        return {
            "pulse_beta": cluster_outputs["cluster_seedness_beta"] * node_gate,
            "pulse_x_c": cluster_outputs["latent_cluster_coordinate_z"]
            * gate.unsqueeze(-1)
            * node_gate.unsqueeze(-1),
            "pulse_score": cluster_outputs["refined_pulse_score"] * node_gate,
            "pulse_time": cluster_outputs["refined_time"] * node_gate,
            "pulse_charge": cluster_outputs["refined_charge"] * node_gate,
            "proposal_score": proposal["pulse_score"] * node_gate,
            "proposal_time": proposal["pulse_time"] * node_gate,
            "proposal_width": proposal["pulse_width"] * node_gate,
            "proposal_amplitude": proposal["pulse_amplitude"] * node_gate,
            "token_mask": cluster_outputs["cluster_token_mask"] & node_mask.unsqueeze(-1),
        }

    def propose_pulses(self, x, pos, fea_mask, node_mask):
        batch_size, num_nodes = node_mask.shape
        token_shape = (batch_size, num_nodes, self.num_pulse_tokens)
        embed = self.weight * torch.ones(
            batch_size, num_nodes, self.num_pulse_tokens, 4, device=x.device
        )
        return {
            "pulse_embedding": embed,
            "pulse_score": torch.sigmoid(self.weight) * torch.ones(token_shape, device=x.device),
            "pulse_time": 0.5 * torch.ones(token_shape, device=x.device),
            "pulse_width": torch.ones(token_shape, device=x.device),
            "pulse_amplitude": torch.ones(token_shape, device=x.device),
            "token_mask": node_mask.unsqueeze(-1).expand_as(
                torch.ones(token_shape, dtype=torch.bool, device=x.device)
            ),
            "pos": pos,
            "base_token_time": 0.5 * torch.ones(token_shape, device=x.device),
        }

    def cluster_pulses(self, proposal, prune_mask=None, soft_pruning=True):
        token_mask = proposal["token_mask"]
        batch_size, num_nodes, num_tokens = token_mask.shape
        gate = token_mask.to(dtype=proposal["pulse_score"].dtype)
        return {
            "cluster_seedness_beta": torch.sigmoid(self.weight)
            * torch.ones(batch_size, num_nodes, num_tokens, device=gate.device),
            "latent_cluster_coordinate_z": self.weight
            * torch.ones(batch_size, num_nodes, num_tokens, 2, device=gate.device),
            "refined_pulse_score": torch.sigmoid(self.weight)
            * torch.ones(batch_size, num_nodes, num_tokens, device=gate.device),
            "refined_time": 0.5 * torch.ones(batch_size, num_nodes, num_tokens, device=gate.device),
            "refined_charge": torch.ones(batch_size, num_nodes, num_tokens, device=gate.device),
            "cluster_token_mask": token_mask,
            "cluster_token_gate": gate,
        }

def build_waveform_trainer(tmp_path, dataloader, **extra_config):
    model = DummyModel(input_type="waveform")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = {
        "epochs": 1,
        "save_dir": str(tmp_path / "saved"),
        "log_dir": str(tmp_path / "logs"),
        **extra_config,
    }
    return WaveformOCTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=torch.device("cpu"),
        dataloader=dataloader,
        valid_dataloader=None,
        lr_scheduler=None,
        logger=None,
    )


def build_multi_pulse_trainer(tmp_path, dataloader, **extra_config):
    model = DummyMultiPulseModel(input_type="waveform", num_pulse_tokens=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = {
        "epochs": 1,
        "save_dir": str(tmp_path / "saved"),
        "log_dir": str(tmp_path / "logs"),
        **extra_config,
    }
    return MultiPulseOCTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=torch.device("cpu"),
        dataloader=dataloader,
        valid_dataloader=None,
        lr_scheduler=None,
        logger=None,
    )


def test_create_sample_mask_keeps_signal_only_graphs():
    object_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    batch = torch.tensor([0, 0, 0], dtype=torch.long)

    mask = create_sample_mask(object_ids, batch=batch, scale=5.0, bkg_id=-1)

    assert torch.equal(mask, torch.tensor([True, True, True]))


def test_waveform_trainer_accepts_legacy_config_keys(tmp_path):
    cfg_path = tmp_path / "vme.csv"
    cfg_path.write_text("channel,FADC250_ALLCH_PED\n0,1.5\n1,2.5\n")

    trainer = build_waveform_trainer(
        tmp_path,
        dataloader=[],
        vme_config=str(cfg_path),
    )

    assert "FADC250_ALLCH_PED" in trainer.model.vme_config
    assert torch.equal(
        trainer.model.vme_config["FADC250_ALLCH_PED"],
        torch.tensor([1.5, 2.5], dtype=torch.float32),
    )


def test_waveform_preprocess_checks_channel_indices(tmp_path):
    trainer = build_waveform_trainer(tmp_path, dataloader=[])
    trainer.model.vme_config = {"FADC250_ALLCH_PED": torch.tensor([0.5, 1.5])}

    wf = torch.ones(1, 4)
    channels = torch.tensor([2])

    try:
        trainer.model._preprocess_wf(wf, channels)
    except ValueError as exc:
        assert "Waveform channels must be in" in str(exc)
    else:
        raise AssertionError("Expected out-of-range channel indices to raise ValueError")


def test_export_onnx_uses_preprocessed_features_and_missing_batch_defaults(tmp_path, monkeypatch):
    x = torch.tensor([[10.0, 11.0], [20.0, 21.0]], dtype=torch.float32)
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    data = DummyData(x=x, pos=pos, y=y)

    cfg_path = tmp_path / "vme.csv"
    cfg_path.write_text("channel,FADC250_ALLCH_PED\n0,1.0\n1,2.0\n")
    trainer = build_waveform_trainer(
        tmp_path,
        dataloader=[data],
        vme_config=str(cfg_path),
    )

    captured = {}

    def fake_export(model, args, path, **kwargs):
        captured["x"] = args[0].clone()
        captured["pos"] = args[1].clone()
        captured["fea_mask"] = args[2].clone()
        captured["node_mask"] = args[3].clone()
        captured["path"] = path

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    trainer.export_onnx(tmp_path / "model.onnx")

    expected_x = torch.tensor([[[9.0, 10.0], [18.0, 19.0]]], dtype=torch.float32)
    assert torch.equal(captured["x"], expected_x)
    assert captured["pos"].shape == (1, 2, 2)
    assert torch.equal(captured["node_mask"], torch.tensor([[True, True]]))
    assert torch.equal(captured["fea_mask"], torch.tensor([[[True, True], [True, True]]]))


def test_multi_pulse_trainer_computes_loss_dict(tmp_path):
    x = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float32)
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([[3, -1], [-1, -1]], dtype=torch.long)
    data = DummyData(x=x, pos=pos, y=y)

    trainer = build_multi_pulse_trainer(
        tmp_path,
        dataloader=[data],
        noise_idx=-1,
    )

    outputs = trainer._forward_losses(data)

    assert outputs["loss"].ndim == 0
    assert outputs["proposal_score_loss"].ndim == 0
    assert outputs["refined_score_loss"].ndim == 0
    assert outputs["beta"].shape == (2,)


def test_multi_pulse_trainer_rejects_target_width_larger_than_model(tmp_path):
    x = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float32)
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([[3, 4, -1], [-1, -1, -1]], dtype=torch.long)
    data = DummyData(x=x, pos=pos, y=y)

    trainer = build_multi_pulse_trainer(
        tmp_path,
        dataloader=[data],
        noise_idx=-1,
    )

    try:
        trainer._forward_losses(data)
    except ValueError as exc:
        assert "Target object-id width exceeds model pulse-token width" in str(exc)
    else:
        raise AssertionError("Expected multi-pulse target width mismatch to raise ValueError")
