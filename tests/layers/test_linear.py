import torch
from bensemble.layers import BayesianLinear


def test_linear_shape(input_data):
    layer = BayesianLinear(10, 5)
    out = layer(input_data)
    assert out.shape == (input_data.shape[0], 5)


def test_linear_kl_calculation():
    layer = BayesianLinear(10, 5)
    kl = layer.kl_divergence()
    assert isinstance(kl, torch.Tensor)
    assert kl.item() > 0


def test_stochastic_behavior(input_data):
    layer = BayesianLinear(10, 5)
    layer.train()

    out1 = layer(input_data)
    out2 = layer(input_data)

    assert not torch.allclose(out1, out2)


def test_deterministic_behavior(input_data):
    layer = BayesianLinear(10, 5)
    layer.eval()

    out1 = layer(input_data)
    out2 = layer(input_data)

    assert torch.allclose(out1, out2)


def test_gradients_flow(input_data):
    layer = BayesianLinear(10, 5)
    out = layer(input_data)
    loss = out.sum()
    loss.backward()

    assert layer.w_mu.grad is not None
    assert layer.w_rho.grad is not None
