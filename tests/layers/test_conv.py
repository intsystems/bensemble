import torch
import torch.nn as nn

from bensemble.layers.conv import BayesianConv2d


def test_conv_output_shape_train():
    """Train mode produces the expected spatial output shape."""
    layer = BayesianConv2d(1, 4, 3)
    layer.train()
    x = torch.randn(2, 1, 8, 8)
    out = layer(x)
    assert out.shape == (2, 4, 6, 6)


def test_conv_output_shape_eval():
    """Eval mode produces the expected spatial output shape."""
    layer = BayesianConv2d(1, 4, 3)
    layer.eval()
    x = torch.randn(2, 1, 8, 8)
    out = layer(x)
    assert out.shape == (2, 4, 6, 6)


def test_conv_stochastic_train():
    """Train mode produces different outputs per forward pass."""
    torch.manual_seed(0)
    layer = BayesianConv2d(1, 4, 3)
    layer.train()
    x = torch.randn(2, 1, 8, 8)
    out1 = layer(x)
    out2 = layer(x)
    assert not torch.allclose(out1, out2)


def test_conv_deterministic_eval():
    """Eval mode produces identical outputs per forward pass."""
    layer = BayesianConv2d(1, 4, 3)
    layer.eval()
    x = torch.randn(2, 1, 8, 8)
    out1 = layer(x)
    out2 = layer(x)
    assert torch.allclose(out1, out2)


def test_conv_kl_positive():
    """KL divergence is a positive tensor."""
    layer = BayesianConv2d(1, 4, 3)
    kl = layer.kl_divergence()
    assert isinstance(kl, torch.Tensor)
    assert kl.item() > 0


def test_conv_tuple_kernel_size():
    """Tuple kernel size produces the correct asymmetric output shape."""
    layer = BayesianConv2d(1, 4, kernel_size=(3, 5))
    layer.train()
    x = torch.randn(1, 1, 8, 8)
    out = layer(x)
    assert out.shape == (1, 4, 6, 4)


def test_conv_with_padding():
    """Padding preserves the spatial dimensions."""
    layer = BayesianConv2d(1, 4, 3, padding=1)
    layer.train()
    x = torch.randn(1, 1, 8, 8)
    out = layer(x)
    assert out.shape == (1, 4, 8, 8)


def test_conv_gradients_flow():
    """Gradients flow to both weight mean and rho parameters."""
    layer = BayesianConv2d(1, 4, 3)
    layer.train()
    x = torch.randn(2, 1, 8, 8)
    out = layer(x)
    out.sum().backward()
    assert layer.w_mu.grad is not None
    assert layer.w_rho.grad is not None


def test_conv_reset_parameters():
    """reset_parameters zeros the bias mean."""
    layer = BayesianConv2d(1, 4, 3)
    layer.reset_parameters()
    assert torch.allclose(layer.b_mu, torch.zeros(layer.out_channels))
