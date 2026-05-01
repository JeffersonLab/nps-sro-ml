import torch
from torch import nn

from models.oc_base import ObjectCondensationBaseModel
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
