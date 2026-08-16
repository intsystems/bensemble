import math

import torch
from torch import nn

from bensemble.layers import BayesianLinear
from bensemble.layers.base import BaseBayesianLayer
from bensemble.layers.conv import BayesianConv2d

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


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
        model: Target neural network containing nn.Dropout layers.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
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


def predict_with_uncertainty(
    model: nn.Module, x: torch.Tensor, num_samples: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimates predictive mean and standard deviation via Monte Carlo sampling.

    Args:
        model: Neural network model containing stochastic layers.
        x: Input tensor.
        num_samples: Number of forward Monte Carlo passes. Defaults to 100.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple (mean, std) representing
        the predictive mean and unbiased sample standard deviation.
    """
    was_training = model.training
    model.eval()

    for module in model.modules():
        if isinstance(module, (BayesianLinear, BayesianConv2d)):
            module.train()

    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(num_samples)])

    mean = preds.mean(dim=0)
    std = preds.std(dim=0, unbiased=True)

    model.train(was_training)

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
