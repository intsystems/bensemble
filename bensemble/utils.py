import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.func import functional_call

from bensemble.layers import BayesianLinear
from bensemble.layers.base import BaseBayesianLayer
from bensemble.layers.conv import BayesianConv2d

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


@contextmanager
def _preserve_training_modes(model: nn.Module) -> Iterator[None]:
    """Restores every module's training flag, including after an exception."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Computes the standard normal probability density function phi(x).

    Args:
        x: Input tensor.

    Returns:
        torch.Tensor: Probability density evaluated at x.
    """
    return torch.exp(-0.5 * x * x) * _INV_SQRT_2PI


def standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Computes the standard normal cumulative distribution function Phi(x).

    Args:
        x: Input tensor.

    Returns:
        torch.Tensor: Cumulative distribution value evaluated at x.
    """
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def enable_dropout(model: nn.Module) -> None:
    """Enables dropout layers during evaluation for Monte Carlo Dropout.

    Args:
        model: Target neural network containing PyTorch dropout layers.
    """
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.train()


def get_total_kl(model: nn.Module) -> torch.Tensor:
    """Calculates the sum of KL divergences of all Bayesian layers in the model.

    Args:
        model: Neural network model containing Bayesian layers.

    Returns:
        torch.Tensor: Total accumulated KL divergence scalar.
    """
    total_kl = 0.0

    for module in model.modules():
        if hasattr(module, "kl_divergence"):
            total_kl += module.kl_divergence()

    return total_kl


def sample_with_log_probs(
    model: nn.Module, x: torch.Tensor, num_samples: int = 10
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draws model weights and returns predictions with their log densities.

    Each draw uses one set of weights across the batch and every call to a reused
    layer. Unlike local reparameterization, these predictions have matching
    parameter-space prior and posterior densities for a Rényi objective.
    Model parameters, buffers, and training modes are preserved; BatchNorm
    running statistics are not updated.

    Args:
        model: Model containing BayesianLinear or BayesianConv2d layers.
            Other layers retain their training modes; set stochastic layers such
            as dropout to eval if only Bayesian weight uncertainty is wanted.
        x: Input batch.
        num_samples: Positive number of weight draws.

    Returns:
        Tuple of predictions shaped (K, batch, ...), log prior densities (K,),
        and log posterior densities (K,). All three retain gradients.

    Raises:
        ValueError: If the sample count is invalid, no supported layers exist,
            or distinct layers share posterior means.
        TypeError: If the model contains an unsupported Bayesian layer.
    """
    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, int)
        or num_samples < 1
    ):
        raise ValueError("num_samples must be a positive integer.")

    layers = []
    parameters = []
    sampled_means = set()
    parameter_uses = {}
    for layer in model.modules():
        for parameter in layer.parameters(recurse=False):
            key = id(parameter)
            parameter_uses[key] = parameter_uses.get(key, 0) + 1
    for name, layer in model.named_modules():
        if not isinstance(layer, BaseBayesianLayer):
            continue
        if not isinstance(layer, (BayesianLinear, BayesianConv2d)):
            raise TypeError(
                "Weight sampling supports BayesianLinear and BayesianConv2d."
            )
        layers.append((layer, layer.training))
        prefix = f"{name}." if name else ""
        for stem in ("w", "b"):
            mu = getattr(layer, f"{stem}_mu")
            if id(mu) in sampled_means or parameter_uses[id(mu)] > 1:
                raise ValueError(
                    "Tied weights across distinct layers are not supported."
                )
            sampled_means.add(id(mu))
            parameters.append(
                (
                    f"{prefix}{stem}_mu",
                    mu,
                    getattr(layer, f"{stem}_rho"),
                    layer.prior_sigma,
                )
            )
    if not layers:
        raise ValueError(
            "The model must contain BayesianLinear or BayesianConv2d layers."
        )

    predictions, log_priors, log_posteriors = [], [], []
    try:
        for layer, _ in layers:
            layer.training = False
        for _ in range(num_samples):
            sampled_parameters = {}
            prior_terms, posterior_terms = [], []
            for name, mu, rho, prior_sigma in parameters:
                posterior = Normal(mu, F.softplus(rho))
                sample = posterior.rsample()
                prior = Normal(mu.new_zeros(()), mu.new_tensor(prior_sigma))
                sampled_parameters[name] = sample
                prior_terms.append(prior.log_prob(sample).sum())
                posterior_terms.append(posterior.log_prob(sample).sum())
            buffers = {name: value.clone() for name, value in model.named_buffers()}
            predictions.append(
                functional_call(
                    model, (sampled_parameters, buffers), (x,), tie_weights=False
                )
            )
            log_priors.append(torch.stack(prior_terms).sum())
            log_posteriors.append(torch.stack(posterior_terms).sum())
    finally:
        for layer, training in layers:
            layer.training = training

    return (
        torch.stack(predictions),
        torch.stack(log_priors),
        torch.stack(log_posteriors),
    )


@torch.no_grad()
def _mc_predictions(
    model: nn.Module, x: torch.Tensor, num_samples: int
) -> torch.Tensor:
    """Draw one convolution kernel and bias per layer and model forward."""
    convs = [
        layer
        for layer in model.modules()
        if isinstance(layer, BayesianConv2d) and layer.training
    ]
    previous = [
        (getattr(layer, "_mc_weight", None), getattr(layer, "_mc_bias", None))
        for layer in convs
    ]
    predictions = []
    try:
        for _ in range(num_samples):
            for layer in convs:
                layer._draw_mc_weights()
            predictions.append(model(x))
    finally:
        for layer, (weight, bias) in zip(convs, previous):
            layer._mc_weight, layer._mc_bias = weight, bias
    return torch.stack(predictions)


def predict_with_uncertainty(
    model: nn.Module, x: torch.Tensor, num_samples: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimates predictive mean and standard deviation via Monte Carlo sampling.

    Restores each layer's original training flag, including on errors.

    Args:
        model: Neural network model containing stochastic layers.
        x: Input tensor.
        num_samples: Number of forward Monte Carlo passes. Defaults to 100.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple (mean, std) representing
        the predictive mean and unbiased sample standard deviation.
    """
    with _preserve_training_modes(model):
        model.eval()

        for module in model.modules():
            if isinstance(module, (BayesianLinear, BayesianConv2d)):
                module.train()

        preds = _mc_predictions(model, x, num_samples)

        mean = preds.mean(dim=0)
        std = preds.std(dim=0, unbiased=True)
        return mean, std


def prune_model(model: torch.nn.Module, threshold: float = 0.83) -> float:
    """Applies Graves' SNR-based weight pruning to all Bayesian layers in the model.

    Args:
        model: Neural network containing BaseBayesianLayer modules.
        threshold: Signal-to-Noise Ratio (SNR) pruning threshold. Defaults to 0.83.

    Returns:
        float: Overall sparsity ratio of pruned weights across all Bayesian layers (0.0 to 1.0).
    """
    total_weights = 0
    total_pruned = 0

    for module in model.modules():
        if isinstance(module, BaseBayesianLayer):
            masks = module.get_pruning_masks(threshold)
            for mask in masks.values():
                total_weights += mask.numel()
                total_pruned += (mask == 0.0).sum().item()

            module.apply_pruning(threshold)

    overall_sparsity = total_pruned / total_weights if total_weights > 0 else 0.0
    return overall_sparsity
