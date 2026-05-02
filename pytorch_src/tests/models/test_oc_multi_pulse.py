import torch

from models.oc_multi_pulse import MultiPulseObjectCondensationModel


def test_oc_multi_pulse_waveform_shapes_and_proposal_cache():
    batch_size, num_nodes, waveform_len = 2, 5, 110
    x = torch.randn(batch_size, num_nodes, waveform_len)
    pos = torch.randn(num_nodes, 2)
    fea_mask = torch.ones(batch_size, num_nodes, waveform_len, dtype=torch.bool)
    node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

    model = MultiPulseObjectCondensationModel(
        input_type="waveform",
        d_model=32,
        wf_d_model=32,
        wf_enc_layers=1,
        wf_enc_heads=4,
        n_enc_layers=1,
        num_heads=4,
        num_pulse_tokens=2,
    )

    x_c, beta = model(x, pos, fea_mask, node_mask)

    assert x_c.shape == (batch_size, num_nodes, 2)
    assert beta.shape == (batch_size, num_nodes, 1)
    assert model.last_proposal_score.shape == (batch_size, num_nodes, 2)
    assert model.last_proposal_time.shape == (batch_size, num_nodes, 2)
    assert model.last_proposal_width.shape == (batch_size, num_nodes, 2)
    assert model.last_proposal_amplitude.shape == (batch_size, num_nodes, 2)
    assert model.last_proposal_embedding.shape == (batch_size, num_nodes, 2, 32)
    assert model.last_token_beta.shape == (batch_size, num_nodes, 2)
    assert model.last_cluster_z.shape == (batch_size, num_nodes, 2, 2)
    assert model.last_refined_score.shape == (batch_size, num_nodes, 2)
    assert model.last_refined_time.shape == (batch_size, num_nodes, 2)
    assert model.last_refined_charge.shape == (batch_size, num_nodes, 2)


def test_oc_multi_pulse_projects_block_features_when_dims_differ():
    batch_size, num_nodes, waveform_len = 2, 5, 110
    x = torch.randn(batch_size, num_nodes, waveform_len)
    pos = torch.randn(num_nodes, 2)
    fea_mask = torch.ones(batch_size, num_nodes, waveform_len, dtype=torch.bool)
    node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

    model = MultiPulseObjectCondensationModel(
        input_type="waveform",
        d_model=64,
        wf_d_model=32,
        wf_enc_layers=1,
        wf_enc_heads=4,
        n_enc_layers=1,
        num_heads=4,
        num_pulse_tokens=2,
    )

    proposal = model.propose_pulses(x, pos, fea_mask, node_mask)

    assert proposal["pulse_embedding"].shape == (batch_size, num_nodes, 2, 64)
    assert proposal["pulse_score"].shape == (batch_size, num_nodes, 2)


def test_oc_multi_pulse_masks_invalid_nodes():
    batch_size, num_nodes, waveform_len = 1, 4, 110
    x = torch.randn(batch_size, num_nodes, waveform_len)
    pos = torch.randn(num_nodes, 2)
    fea_mask = torch.ones(batch_size, num_nodes, waveform_len, dtype=torch.bool)
    node_mask = torch.tensor([[True, True, False, False]])

    model = MultiPulseObjectCondensationModel(
        input_type="waveform",
        d_model=32,
        wf_d_model=32,
        wf_enc_layers=1,
        wf_enc_heads=4,
        n_enc_layers=1,
        num_heads=4,
        num_pulse_tokens=2,
    )

    x_c, beta = model(x, pos, fea_mask, node_mask)

    assert torch.allclose(x_c[:, 2:], torch.zeros_like(x_c[:, 2:]))
    assert torch.allclose(beta[:, 2:], torch.zeros_like(beta[:, 2:]))
    assert torch.allclose(
        model.last_proposal_score[:, 2:],
        torch.zeros_like(model.last_proposal_score[:, 2:]),
    )
    assert torch.allclose(
        model.last_token_beta[:, 2:],
        torch.zeros_like(model.last_token_beta[:, 2:]),
    )


def test_oc_multi_pulse_exposes_hard_pruning_path():
    batch_size, num_nodes, waveform_len = 1, 6, 110
    x = torch.randn(batch_size, num_nodes, waveform_len)
    pos = torch.randn(num_nodes, 2)
    fea_mask = torch.ones(batch_size, num_nodes, waveform_len, dtype=torch.bool)
    node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

    model = MultiPulseObjectCondensationModel(
        input_type="waveform",
        d_model=32,
        wf_d_model=32,
        wf_enc_layers=1,
        wf_enc_heads=4,
        n_enc_layers=1,
        num_heads=4,
        num_pulse_tokens=3,
    )

    proposal = model.propose_pulses(x, pos, fea_mask, node_mask)
    prune_mask = model.build_pruning_mask(
        proposal["pulse_score"],
        proposal["token_mask"],
        score_threshold=0.5,
        top_m=2,
    )
    cluster = model.cluster_pulses(
        proposal=proposal,
        prune_mask=prune_mask,
        soft_pruning=False,
    )

    assert prune_mask.shape == (batch_size, num_nodes, 3)
    assert cluster["cluster_seedness_beta"].shape == (batch_size, num_nodes, 3)
    assert cluster["latent_cluster_coordinate_z"].shape == (batch_size, num_nodes, 3, 2)
    assert cluster["cluster_token_mask"].shape == (batch_size, num_nodes, 3)
