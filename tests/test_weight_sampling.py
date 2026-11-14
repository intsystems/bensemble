import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from bensemble.layers import BayesianConv2d, BayesianLinear
from bensemble.layers.base import BaseBayesianLayer
from bensemble.utils import sample_with_log_probs


@pytest.mark.parametrize("convolution", [False, True])
def test_predictions_and_log_densities_use_the_same_weights(convolution):
    torch.manual_seed(4)
    layer = (
        BayesianConv2d(2, 3, kernel_size=1, prior_sigma=1.7)
        if convolution
        else BayesianLinear(2, 3, prior_sigma=1.7)
    ).double()
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    if convolution:
        inputs = inputs[:, :, None, None]

    predictions, log_prior, log_posterior = sample_with_log_probs(layer, inputs, 5)

    expected_shape = (5, 3, 3, 1, 1) if convolution else (5, 3, 3)
    assert predictions.shape == expected_shape
    assert log_prior.shape == log_posterior.shape == (5,)
    for index, prediction in enumerate(predictions):
        affine_outputs = prediction.reshape(3, 3)
        bias = affine_outputs[0]
        weight = (affine_outputs[1:] - bias).T.reshape_as(layer.w_mu)
        expected_prior = sum(
            Normal(0.0, layer.prior_sigma).log_prob(value).sum()
            for value in (weight, bias)
        )
        expected_posterior = sum(
            Normal(mu, F.softplus(rho)).log_prob(value).sum()
            for value, mu, rho in (
                (weight, layer.w_mu, layer.w_rho),
                (bias, layer.b_mu, layer.b_rho),
            )
        )
        torch.testing.assert_close(log_prior[index], expected_prior)
        torch.testing.assert_close(log_posterior[index], expected_posterior)


def test_global_weight_draw_is_shared_across_identical_inputs():
    model = nn.Sequential(BayesianLinear(2, 4), nn.Tanh(), BayesianLinear(4, 1))
    inputs = torch.tensor([[0.3, -0.7]]).repeat(6, 1)

    predictions, _, _ = sample_with_log_probs(model, inputs, 5)

    torch.testing.assert_close(predictions, predictions[:, :1].expand_as(predictions))
    assert predictions[:, 0].std() > 0


def test_reused_layer_uses_one_draw_and_one_density():
    class SharedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = BayesianLinear(2, 2).double()
            self.second = self.first

        def forward(self, inputs):
            return self.second(torch.tanh(self.first(inputs)))

    model = SharedModel()
    parameters = dict(model.named_parameters())
    calls = []
    handle = model.first.register_forward_hook(
        lambda layer, inputs, output: calls.append(
            (layer.w_mu.detach().clone(), layer.b_mu.detach().clone())
        )
    )
    inputs = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    try:
        predictions, log_prior, log_posterior = sample_with_log_probs(model, inputs, 3)
    finally:
        handle.remove()

    assert len(calls) == 6
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]
    for index in range(3):
        weight, bias = calls[2 * index]
        torch.testing.assert_close(weight, calls[2 * index + 1][0])
        torch.testing.assert_close(bias, calls[2 * index + 1][1])
        expected = F.linear(torch.tanh(F.linear(inputs, weight, bias)), weight, bias)
        torch.testing.assert_close(predictions[index], expected)
        torch.testing.assert_close(
            log_prior[index],
            Normal(0.0, 1.0).log_prob(weight).sum()
            + Normal(0.0, 1.0).log_prob(bias).sum(),
        )
        torch.testing.assert_close(
            log_posterior[index],
            Normal(model.first.w_mu, F.softplus(model.first.w_rho))
            .log_prob(weight)
            .sum()
            + Normal(model.first.b_mu, F.softplus(model.first.b_rho))
            .log_prob(bias)
            .sum(),
        )


def test_weight_sampling_retains_parameter_gradients_and_model_state():
    model = nn.Sequential(
        BayesianConv2d(1, 2, kernel_size=1),
        nn.Flatten(),
        nn.Tanh(),
        BayesianLinear(8, 1),
    ).double()
    model.train()
    model[0].eval()
    parameters = dict(model.named_parameters())
    values = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [layer.training for layer in model.modules()]

    predictions, log_prior, log_posterior = sample_with_log_probs(
        model, torch.randn(3, 1, 2, 2, dtype=torch.float64), 4
    )
    prediction_gradients = torch.autograd.grad(
        predictions.square().sum(), tuple(model.parameters()), retain_graph=True
    )
    assert all(torch.isfinite(gradient).all() for gradient in prediction_gradients)
    assert all(gradient.abs().sum() > 0 for gradient in prediction_gradients)
    assert log_prior.requires_grad and log_posterior.requires_grad
    loss = predictions.square().sum() + (log_posterior - log_prior).mean()
    loss.backward()

    assert [layer.training for layer in model.modules()] == modes
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]
        torch.testing.assert_close(parameter, values[name])
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_weight_sampling_restores_parameters_and_modes_after_error():
    class Fail(nn.Module):
        def forward(self, inputs):
            raise RuntimeError("forward failed")

    model = nn.Sequential(BayesianLinear(2, 3), nn.BatchNorm1d(3), Fail())
    model.train()
    model[2].eval()
    parameters = dict(model.named_parameters())
    values = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [layer.training for layer in model.modules()]

    with pytest.raises(RuntimeError, match="forward failed"):
        sample_with_log_probs(model, torch.ones(2, 2))

    assert [layer.training for layer in model.modules()] == modes
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]
        torch.testing.assert_close(parameter, values[name])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, values[name])


@pytest.mark.parametrize("num_samples", [0, -1, 1.5, True])
def test_weight_sampling_rejects_invalid_sample_count(num_samples):
    with pytest.raises(ValueError, match="positive integer"):
        sample_with_log_probs(BayesianLinear(1, 1), torch.ones(1, 1), num_samples)


def test_weight_sampling_requires_supported_bayesian_layers():
    with pytest.raises(ValueError, match="must contain"):
        sample_with_log_probs(nn.Linear(1, 1), torch.ones(1, 1))
    with pytest.raises(TypeError, match="supports BayesianLinear and BayesianConv2d"):
        sample_with_log_probs(BaseBayesianLayer(), torch.ones(1, 1))


def test_weight_sampling_preserves_batchnorm_buffers_and_mode():
    model = nn.Sequential(BayesianLinear(2, 3), nn.BatchNorm1d(3))
    model.train()
    buffers = {name: value.clone() for name, value in model.named_buffers()}

    predictions, _, _ = sample_with_log_probs(model, torch.randn(4, 2), 3)

    assert torch.isfinite(predictions).all()
    assert all(layer.training for layer in model.modules())
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, buffers[name])


def test_weight_sampling_rejects_tied_parameters_in_distinct_layers():
    model = nn.Sequential(BayesianLinear(2, 2), BayesianLinear(2, 2))
    model[1].w_mu = model[0].w_mu
    model[1].w_rho = model[0].w_rho

    with pytest.raises(ValueError, match="Tied weights"):
        sample_with_log_probs(model, torch.ones(1, 2))


def test_weight_sampling_rejects_mean_shared_with_deterministic_layer():
    model = nn.Sequential(BayesianLinear(2, 2), nn.Linear(2, 2))
    model[1].weight = model[0].w_mu

    with pytest.raises(ValueError, match="Tied weights"):
        sample_with_log_probs(model, torch.ones(1, 2))
