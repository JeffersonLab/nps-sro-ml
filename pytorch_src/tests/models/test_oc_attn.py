import torch

from models.oc_attn import ObjectCondensationModel


def test_oc_attn_pulse_set_embed_out_projection():
    """Pulse-set path should work even when embed_out != d_model."""
    batch_size, num_nodes, num_pulses = 2, 5, 4
    x = torch.randn(batch_size, num_nodes, num_pulses, 2)
    pos = torch.randn(batch_size, num_nodes, 2)
    fea_mask = torch.ones(batch_size, num_nodes, num_pulses, dtype=torch.bool)
    node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

    model = ObjectCondensationModel(
        input_type="pulse_set",
        embed_in=8,
        embed_out=16,
        d_model=32,
        n_enc_layers=1,
        num_heads=4,
    )

    x_c, beta = model(x, pos, fea_mask, node_mask)

    assert x_c.shape == (batch_size, num_nodes, 2)
    assert beta.shape == (batch_size, num_nodes, 1)

def test_oc_attn_waveform_default_ff_width_matches_encoder_contract():
    """Waveform defaults should keep the encoder FF width at 4x d_model, not d_model squared."""
    model = ObjectCondensationModel(
        input_type="waveform",
        wf_d_model=32,
        wf_enc_layers=1,
        wf_enc_heads=4,
        d_model=32,
        n_enc_layers=1,
        num_heads=4,
    )

    first_linear = model.fea_encoder.layers[0].feedforward[0]
    assert first_linear.in_features == 32
    assert first_linear.out_features == 128
