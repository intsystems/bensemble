import pytest
import torch
from torch import nn

from bensemble.core.ensemble import Ensemble
from bensemble.layers import BayesianConv2d
from bensemble.utils import predict_with_uncertainty


def test_conv_members_draw_one_kernel_and_bias_per_channel():
    torch.manual_seed(7)
    conv = BayesianConv2d(1, 2, kernel_size=1, init_sigma=1.0)
    ensemble = Ensemble.from_stochastic(conv, num_samples=8, mode="bayesian")
    # For x=[0, 1, 2] the outputs must be [b, b+w, b+2w] in both images.
    x = torch.arange(3.0).reshape(1, 1, 1, 3).repeat(2, 1, 1, 1)
    draws = ensemble.predict_members(x)
    torch.testing.assert_close(draws[:, 0], draws[:, 1])
    bias, middle, last = draws[:, 0, :, 0].unbind(dim=-1)
    weight = middle - bias
    torch.testing.assert_close(last, bias + 2 * weight)
    assert (bias.std(dim=0) > 0).all()
    assert (weight.std(dim=0) > 0).all()
    assert not torch.equal(bias[:, 0], bias[:, 1])


def test_pooled_conv_retains_bias_variance():
    torch.manual_seed(19)
    model = nn.Sequential(
        BayesianConv2d(1, 1, kernel_size=1, init_sigma=1.0),
        nn.AdaptiveAvgPool2d(1),
    )
    # Every pixel of a zero image equals the same bias, even after averaging.
    _, std = predict_with_uncertainty(model, torch.zeros(1, 1, 8, 8), num_samples=256)
    assert std.item() == pytest.approx(1.0, abs=0.2)


def test_reused_conv_draws_the_same_weights_in_one_member():
    class ReusedConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = BayesianConv2d(1, 1, kernel_size=1)

        def forward(self, x):
            return self.conv(x) - self.conv(x)

    ensemble = Ensemble.from_stochastic(ReusedConv(), num_samples=4)
    draws = ensemble.predict_members(torch.ones(1, 1, 3, 3))
    assert torch.count_nonzero(draws) == 0


def test_failed_mc_forward_restores_local_noise_for_training():
    conv = BayesianConv2d(1, 1, kernel_size=1).train()
    x = torch.zeros(1, 1, 4, 4)

    def fail(layer, inputs, output):
        torch.testing.assert_close(output, output[..., :1, :1].expand_as(output))
        raise RuntimeError("forward failed")

    with (
        conv.register_forward_hook(fail),
        pytest.raises(RuntimeError, match="forward failed"),
    ):
        predict_with_uncertainty(conv, x, num_samples=2)

    assert conv.training
    assert conv(x).var() > 0
