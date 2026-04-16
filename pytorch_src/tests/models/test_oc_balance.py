import torch

from models.oc_balance import BalancedObjectCondensationModel


def test_oc_balance_pulse_set_forward_shapes():
    batch_size, num_nodes, num_pulses = 2, 7, 4
    x = torch.randn(batch_size, num_nodes, num_pulses, 2)
    pos = torch.randint(0, 10, (batch_size, num_nodes, 2)).float()
    fea_mask = torch.ones(batch_size, num_nodes, num_pulses, dtype=torch.bool)
    node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

    model = BalancedObjectCondensationModel(
        input_type="pulse_set",
        grid_rows=36,
        grid_cols=30,
        embed_in=8,
        embed_out=24,
        d_model=32,
        num_heads=4,
        num_latents=6,
        num_global_layers=1,
    )

    x_c, beta = model(x, pos, fea_mask, node_mask)

    assert x_c.shape == (batch_size, num_nodes, 2)
    assert beta.shape == (batch_size, num_nodes, 1)


def test_oc_balance_masks_padded_nodes():
    batch_size, num_nodes, num_pulses = 1, 5, 3
    x = torch.randn(batch_size, num_nodes, num_pulses, 2)
    pos = torch.tensor([[[0, 0], [0, 1], [1, 1], [2, 2], [3, 3]]], dtype=torch.float32)
    fea_mask = torch.ones(batch_size, num_nodes, num_pulses, dtype=torch.bool)
    node_mask = torch.tensor([[True, True, True, False, False]])

    model = BalancedObjectCondensationModel(
        input_type="pulse_set",
        d_model=32,
        num_heads=4,
        num_latents=4,
        num_global_layers=1,
    )

    x_c, beta = model(x, pos, fea_mask, node_mask)

    assert torch.all(x_c[~node_mask] == 0)
    assert torch.all(beta[~node_mask] == 0)
