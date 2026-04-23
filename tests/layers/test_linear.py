import torch

from bensemble.layers import BayesianLinear


def test_linear_shape(input_data):
    """Output tensor has the expected (batch, out_features) shape."""
    layer = BayesianLinear(10, 5)
    out = layer(input_data)
    assert out.shape == (input_data.shape[0], 5)


def test_linear_kl_calculation():
    """kl_divergence returns a non-negative tensor."""
    layer = BayesianLinear(10, 5)
    kl = layer.kl_divergence()
    assert isinstance(kl, torch.Tensor)
    assert kl.item() >= 0


def test_stochastic_behavior(input_data):
    """Train mode produces different outputs per forward pass."""
    layer = BayesianLinear(10, 5)
    layer.train()

    out1 = layer(input_data)
    out2 = layer(input_data)

    assert not torch.allclose(out1, out2)


def test_deterministic_behavior(input_data):
    """Eval mode produces identical outputs per forward pass."""
    layer = BayesianLinear(10, 5)
    layer.eval()

    out1 = layer(input_data)
    out2 = layer(input_data)

    assert torch.allclose(out1, out2)


def test_gradients_flow(input_data):
    """Gradients flow to both weight mean and rho parameters."""
    layer = BayesianLinear(10, 5)
    out = layer(input_data)
    loss = out.sum()
    loss.backward()

    assert layer.w_mu.grad is not None
    assert layer.w_rho.grad is not None


def test_base_layer_get_pruning_masks():
    """Pruning masks contain only 0 and 1 values for all parameters."""
    layer = BayesianLinear(4, 2)
    masks = layer.get_pruning_masks(threshold=0.0)
    assert "w_mu" in masks and "b_mu" in masks
    for mask in masks.values():
        unique = mask.unique()
        assert all(v in (0.0, 1.0) for v in unique.tolist())


def test_base_layer_snr_dict():
    """SNR values are non-negative for all parameters."""
    layer = BayesianLinear(4, 2)
    snr = layer._get_snr_dict()
    assert "w_mu" in snr and "b_mu" in snr
    for v in snr.values():
        assert (v >= 0).all()


def test_base_layer_compute_kl_for_param():
    """KL divergence for a single parameter pair is non-negative."""
    layer = BayesianLinear(4, 2)
    mu = torch.zeros(3)
    rho = torch.zeros(3)
    kl = layer._compute_kl_for_param(mu, rho)
    assert isinstance(kl, torch.Tensor)
    assert kl.item() >= 0


def test_linear_xavier_init():
    """Xavier initialization produces weights in the expected std range."""
    layer = BayesianLinear(10, 5, weight_init="xavier")
    assert layer.kl_divergence().item() > 0
    # xavier_normal_ on (10→5): theoretical std ≈ sqrt(2/15) ≈ 0.37
    assert 0.1 < layer.w_mu.std().item() < 0.7


def test_linear_normal_init():
    """Normal (Kaiming) initialization produces weights in the expected std range."""
    layer = BayesianLinear(10, 5, weight_init="normal")
    assert layer.kl_divergence().item() > 0
    # kaiming_uniform_ with a=sqrt(5), fan_in=10: theoretical std ≈ 0.18
    assert 0.05 < layer.w_mu.std().item() < 0.35
