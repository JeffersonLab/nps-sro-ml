import pytest
import torch

import models.oc_loss as oc_loss_module
from models.oc_loss import (
    _sum_loss_per_graph,
    oc_attr_loss_per_batch,
    oc_attr_loss_per_graph,
    oc_attr_loss_per_graph_naive,
    oc_coward_loss_per_batch,
    oc_coward_loss_per_graph,
    oc_feat_loss_per_batch,
    oc_feat_loss_per_graph,
    oc_loss_per_batch,
    oc_noise_loss_per_batch,
    oc_noise_loss_per_graph,
    oc_repul_loss_per_batch,
    oc_repul_loss_per_graph,
    oc_repul_loss_per_graph_naive,
)

ATOL = 1e-6


@pytest.fixture
def batch_inputs():
    """Return two graphs with globally unique signal-object IDs."""
    x = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.6, 0.0],
            [1.2, 0.0],
            [0.0, 0.0],
            [0.3, 0.0],
            [0.7, 0.0],
            [1.5, 0.0],
        ],
        dtype=torch.float32,
    )
    beta = torch.tensor(
        [0.90, 0.80, 0.70, 0.20, 0.85, 0.75, 0.65, 0.10],
        dtype=torch.float32,
    )
    object_id = torch.tensor([1, 1, 2, 0, 3, 3, 4, 0], dtype=torch.long)
    is_sig = object_id != 0
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    feat_loss = torch.tensor(
        [0.2, 0.4, 0.3, 1.0, 0.5, 0.1, 0.6, 0.8], dtype=torch.float32
    )
    return x, beta, object_id, is_sig, batch, feat_loss


def _sum_manually(per_graph_fn, batch, *args, **kwargs):
    total = None
    for graph_id in torch.unique(batch, sorted=True):
        mask = batch == graph_id
        graph_args = [
            arg[mask] if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        graph_kwargs = {
            key: value[mask] if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        graph_loss = per_graph_fn(*graph_args, **graph_kwargs)
        total = graph_loss if total is None else total + graph_loss
    return total


def test_sum_loss_per_graph_slices_positional_and_keyword_tensors():
    """Slice every per-node tensor before invoking the graph loss."""
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    weights = torch.tensor([0.5, 1.0, 1.5, 2.0])
    batch = torch.tensor([0, 0, 1, 1])

    def loss_fn(graph_values, *, graph_weights, offset):
        return (graph_values * graph_weights).mean() + offset

    actual = _sum_loss_per_graph(
        loss_fn,
        batch,
        values,
        graph_weights=weights,
        offset=2.0,
    )
    expected = loss_fn(values[:2], graph_weights=weights[:2], offset=2.0)
    expected += loss_fn(values[2:], graph_weights=weights[2:], offset=2.0)

    torch.testing.assert_close(actual, expected)


def test_sum_loss_per_graph_with_no_batch_calls_loss_once():
    """Call the graph loss directly when no batch tensor is supplied."""
    values = torch.tensor([1.0, 2.0, 3.0])
    calls = 0

    def loss_fn(graph_values):
        nonlocal calls
        calls += 1
        return graph_values.sum()

    actual = _sum_loss_per_graph(loss_fn, None, values)

    torch.testing.assert_close(actual, values.sum())
    assert calls == 1


@pytest.mark.parametrize(
    ("vectorized", "naive"),
    [
        (oc_attr_loss_per_graph, oc_attr_loss_per_graph_naive),
        (oc_repul_loss_per_graph, oc_repul_loss_per_graph_naive),
    ],
)
def test_vectorized_per_graph_matches_naive(vectorized, naive):
    """Match each vectorized graph loss to its naive reference."""
    x = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.7, 0.0], [1.1, 0.0], [1.5, 0.0]])
    beta = torch.tensor([0.9, 0.7, 0.8, 0.6, 0.2])
    object_id = torch.tensor([1, 1, 2, 2, 0])
    is_sig = object_id != 0

    actual = vectorized(x, beta, object_id, is_sig, q_min=0.2)
    expected = naive(x, beta, object_id, is_sig, q_min=0.2)

    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=0)


def test_per_graph_scalar_losses_have_expected_values():
    """Compute coward and noise losses with simple exact expectations."""
    beta = torch.tensor([0.9, 0.7, 0.8, 0.3, 0.2])
    object_id = torch.tensor([1, 1, 2, 0, 0])
    is_sig = object_id != 0

    coward = oc_coward_loss_per_graph(beta, object_id, is_sig)
    noise = oc_noise_loss_per_graph(beta, is_sig)

    torch.testing.assert_close(coward, torch.tensor(0.15), atol=ATOL, rtol=0)
    torch.testing.assert_close(noise, torch.tensor(0.25), atol=ATOL, rtol=0)


def test_feature_loss_matches_direct_calculation():
    """Match feature loss to an independently evaluated expression."""
    feat_loss = torch.tensor([0.2, 0.5, 0.8, 1.0])
    beta = torch.tensor([0.8, 0.6, 0.3, 0.2])
    is_sig = torch.tensor([True, True, False, False])
    q_min = 0.2
    q = torch.arctanh(beta).square() + q_min
    expected = (
        (feat_loss[is_sig] * q[is_sig]).sum() + (feat_loss[~is_sig] * q_min).sum()
    ) / q.sum()

    actual = oc_feat_loss_per_graph(feat_loss, beta, is_sig, q_min)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("batch_fn", "graph_fn", "arg_names"),
    [
        (
            oc_attr_loss_per_batch,
            oc_attr_loss_per_graph,
            ("x", "beta", "object_id", "is_sig"),
        ),
        (
            oc_repul_loss_per_batch,
            oc_repul_loss_per_graph,
            ("x", "beta", "object_id", "is_sig"),
        ),
        (
            oc_coward_loss_per_batch,
            oc_coward_loss_per_graph,
            ("beta", "object_id", "is_sig"),
        ),
        (
            oc_noise_loss_per_batch,
            oc_noise_loss_per_graph,
            ("beta", "is_sig"),
        ),
        (
            oc_feat_loss_per_batch,
            oc_feat_loss_per_graph,
            ("feat_loss", "beta", "is_sig"),
        ),
    ],
)
def test_batch_fallback_equals_sum_of_per_graph_losses(
    batch_inputs, batch_fn, graph_fn, arg_names
):
    """Sum per-graph losses identically to every fallback batch wrapper."""
    x, beta, object_id, is_sig, batch, feat_loss = batch_inputs
    values = {
        "x": x,
        "beta": beta,
        "object_id": object_id,
        "is_sig": is_sig,
        "feat_loss": feat_loss,
    }
    graph_args = [values[name] for name in arg_names]
    batch_kwargs = {"batch": batch, "use_scatter": False}
    graph_kwargs = {}
    if batch_fn in (oc_attr_loss_per_batch, oc_repul_loss_per_batch):
        batch_kwargs["q_min"] = 0.2
        graph_kwargs["q_min"] = 0.2
    if batch_fn is oc_repul_loss_per_batch:
        batch_kwargs["margin"] = 0.9
        graph_kwargs["margin"] = 0.9

    if batch_fn is oc_feat_loss_per_batch:
        actual = batch_fn(
            feat_loss,
            beta,
            object_id,
            is_sig,
            batch=batch,
            q_min=0.2,
            use_scatter=False,
        )
        graph_kwargs["q_min"] = 0.2
    else:
        actual = batch_fn(*graph_args, **batch_kwargs)

    expected = _sum_manually(graph_fn, batch, *graph_args, **graph_kwargs)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize(
    ("batch_fn", "graph_fn", "args"),
    [
        (
            oc_attr_loss_per_batch,
            oc_attr_loss_per_graph,
            (
                torch.tensor([[0.0], [0.2], [0.8]]),
                torch.tensor([0.9, 0.7, 0.6]),
                torch.tensor([1, 1, 2]),
                torch.tensor([True, True, True]),
            ),
        ),
        (
            oc_repul_loss_per_batch,
            oc_repul_loss_per_graph,
            (
                torch.tensor([[0.0], [0.2], [0.8]]),
                torch.tensor([0.9, 0.7, 0.6]),
                torch.tensor([1, 1, 2]),
                torch.tensor([True, True, True]),
            ),
        ),
        (
            oc_coward_loss_per_batch,
            oc_coward_loss_per_graph,
            (
                torch.tensor([0.9, 0.7, 0.6]),
                torch.tensor([1, 1, 2]),
                torch.tensor([True, True, True]),
            ),
        ),
        (
            oc_noise_loss_per_batch,
            oc_noise_loss_per_graph,
            (torch.tensor([0.9, 0.7, 0.2]), torch.tensor([True, True, False])),
        ),
    ],
)
def test_batch_none_matches_per_graph(batch_fn, graph_fn, args):
    """Treat a missing batch tensor as one graph."""
    actual = batch_fn(*args, batch=None, use_scatter=False)
    expected = graph_fn(*args)

    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=0)


def test_aggregate_returns_component_losses_and_optional_feature_loss(batch_inputs):
    """Return the same five results as separate batch-loss calls."""
    x, beta, object_id, is_sig, batch, feat_loss = batch_inputs

    actual = oc_loss_per_batch(
        x,
        beta,
        object_id,
        is_sig,
        batch,
        feat_loss=feat_loss,
        q_min=0.2,
        margin=0.9,
        use_scatter=False,
    )
    expected = (
        oc_attr_loss_per_batch(x, beta, object_id, is_sig, batch, q_min=0.2),
        oc_repul_loss_per_batch(
            x, beta, object_id, is_sig, batch, q_min=0.2, margin=0.9
        ),
        oc_coward_loss_per_batch(beta, object_id, is_sig, batch),
        oc_noise_loss_per_batch(beta, is_sig, batch),
        oc_feat_loss_per_batch(feat_loss, beta, object_id, is_sig, batch, q_min=0.2),
    )

    assert len(actual) == 5
    for actual_loss, expected_loss in zip(actual, expected):
        torch.testing.assert_close(actual_loss, expected_loss, atol=ATOL, rtol=0)


def test_aggregate_returns_none_when_feature_loss_is_omitted(batch_inputs):
    """Use None for the optional feature-loss result when it is omitted."""
    x, beta, object_id, is_sig, batch, _ = batch_inputs

    losses = oc_loss_per_batch(x, beta, object_id, is_sig, batch)

    assert len(losses) == 5
    assert losses[-1] is None


@pytest.mark.parametrize(
    "batch_fn",
    [
        oc_attr_loss_per_batch,
        oc_repul_loss_per_batch,
        oc_coward_loss_per_batch,
        oc_noise_loss_per_batch,
        oc_feat_loss_per_batch,
    ],
)
def test_each_batch_function_falls_back_when_scatter_is_unavailable(
    monkeypatch, batch_inputs, batch_fn
):
    """Warn and use the fallback path when scatter is unavailable."""
    x, beta, object_id, is_sig, batch, feat_loss = batch_inputs
    monkeypatch.setattr(oc_loss_module, "_HAS_TORCH_SCATTER", False)

    if batch_fn is oc_attr_loss_per_batch:
        args = (x, beta, object_id, is_sig, batch)
    elif batch_fn is oc_repul_loss_per_batch:
        args = (x, beta, object_id, is_sig, batch)
    elif batch_fn is oc_coward_loss_per_batch:
        args = (beta, object_id, is_sig, batch)
    elif batch_fn is oc_noise_loss_per_batch:
        args = (beta, is_sig, batch)
    else:
        args = (feat_loss, beta, object_id, is_sig, batch)

    expected = batch_fn(*args, use_scatter=False)
    with pytest.warns(UserWarning, match="torch_scatter is not installed"):
        actual = batch_fn(*args, use_scatter=True)

    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=0)


def test_aggregate_falls_back_with_one_warning_when_scatter_is_unavailable(
    monkeypatch, batch_inputs
):
    """Resolve missing scatter once before dispatching aggregate losses."""
    x, beta, object_id, is_sig, batch, _ = batch_inputs
    monkeypatch.setattr(oc_loss_module, "_HAS_TORCH_SCATTER", False)
    expected = oc_loss_per_batch(x, beta, object_id, is_sig, batch, use_scatter=False)

    with pytest.warns(UserWarning, match="torch_scatter is not installed") as warnings:
        actual = oc_loss_per_batch(x, beta, object_id, is_sig, batch, use_scatter=True)

    assert len(warnings) == 1
    for actual_loss, expected_loss in zip(actual[:4], expected[:4]):
        torch.testing.assert_close(actual_loss, expected_loss, atol=ATOL, rtol=0)
    assert actual[-1] is expected[-1] is None


@pytest.mark.skipif(
    not oc_loss_module._HAS_TORCH_SCATTER,
    reason="torch_scatter is not installed",
)
@pytest.mark.parametrize(
    "batch_fn",
    [
        oc_attr_loss_per_batch,
        oc_repul_loss_per_batch,
        oc_coward_loss_per_batch,
        oc_noise_loss_per_batch,
    ],
)
def test_scatter_matches_fallback(batch_inputs, batch_fn):
    """Match installed scatter implementations to fallback results."""
    x, beta, object_id, is_sig, batch, _ = batch_inputs
    if batch_fn is oc_attr_loss_per_batch:
        args = (x, beta, object_id, is_sig, batch)
    elif batch_fn is oc_repul_loss_per_batch:
        args = (x, beta, object_id, is_sig, batch)
    elif batch_fn is oc_coward_loss_per_batch:
        args = (beta, object_id, is_sig, batch)
    else:
        args = (beta, is_sig, batch)

    fallback = batch_fn(*args, use_scatter=False)
    scatter = batch_fn(*args, use_scatter=True)

    torch.testing.assert_close(scatter, fallback, atol=ATOL, rtol=0)


@pytest.mark.parametrize(
    "loss_fn,args",
    [
        (
            oc_attr_loss_per_graph,
            (
                torch.tensor([[0.0], [1.0]]),
                torch.tensor([0.8, 0.7]),
                torch.tensor([0, 0]),
            ),
        ),
        (
            oc_repul_loss_per_graph,
            (
                torch.tensor([[0.0], [1.0]]),
                torch.tensor([0.8, 0.7]),
                torch.tensor([0, 0]),
            ),
        ),
        (
            oc_coward_loss_per_graph,
            (torch.tensor([0.8, 0.7]), torch.tensor([0, 0])),
        ),
    ],
)
def test_object_losses_are_zero_when_there_are_no_signal_nodes(loss_fn, args):
    """Return zero object loss when the signal mask is empty."""
    is_sig = torch.zeros(2, dtype=torch.bool)

    loss = loss_fn(*args, is_sig=is_sig)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_noise_loss_is_zero_when_all_nodes_are_signal():
    """Return zero noise loss when the noise mask is empty."""
    beta = torch.tensor([0.8, 0.7])
    is_sig = torch.ones(2, dtype=torch.bool)

    loss = oc_noise_loss_per_graph(beta, is_sig)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_none_signal_mask_treats_every_node_as_signal():
    """Interpret a missing signal mask as an all-signal mask."""
    beta = torch.tensor([0.8, 0.6, 0.4])
    object_id = torch.tensor([1, 1, 2])
    explicit_mask = torch.ones_like(beta, dtype=torch.bool)

    default = oc_coward_loss_per_graph(beta, object_id)
    explicit = oc_coward_loss_per_graph(beta, object_id, explicit_mask)

    torch.testing.assert_close(default, explicit)


def test_fallback_losses_support_backpropagation(batch_inputs):
    """Propagate finite gradients through the fallback loss paths."""
    x, beta, object_id, is_sig, batch, _ = batch_inputs
    x = x.clone().requires_grad_()
    beta = beta.clone().requires_grad_()

    attr = oc_attr_loss_per_batch(x, beta, object_id, is_sig, batch, use_scatter=False)
    repul = oc_repul_loss_per_batch(
        x, beta, object_id, is_sig, batch, use_scatter=False
    )
    coward = oc_coward_loss_per_batch(beta, object_id, is_sig, batch, use_scatter=False)
    noise = oc_noise_loss_per_batch(beta, is_sig, batch, use_scatter=False)
    (attr + repul + coward + noise).backward()

    assert x.grad is not None
    assert beta.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(beta.grad).all()


@pytest.mark.parametrize(
    "naive_fn",
    [oc_attr_loss_per_graph_naive, oc_repul_loss_per_graph_naive],
)
def test_naive_helpers_reject_autograd_inputs(naive_fn):
    """Reject autograd tensors in reference-only naive helpers."""
    x = torch.tensor([[0.0], [1.0]], requires_grad=True)
    beta = torch.tensor([0.8, 0.7])
    object_id = torch.tensor([1, 1])

    with pytest.raises(RuntimeError, match="testing only"):
        naive_fn(x, beta, object_id)
