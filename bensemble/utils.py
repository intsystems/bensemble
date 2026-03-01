from bensemble.layers.conv import BayesianConv2d
from bensemble.layers import BayesianLinear
import math
from typing import Tuple

import torch
import torch.nn as nn

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal probability density function ϕ(x)"""
    return torch.exp(-0.5 * x * x) * _INV_SQRT_2PI


def standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal cumulative distribution function Φ(x)"""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def enable_dropout(model: nn.Module):
    """Activating dropout layers for MC Dropout"""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def get_total_kl(model: nn.Module) -> torch.Tensor:
    """
    Calculates the sum of KL-divergence of all bayessian layers in the model.
    """
    total_kl = 0.0

    for module in model.modules():
        if hasattr(module, "kl_divergence"):
            total_kl += module.kl_divergence()

    return total_kl


def predict_with_uncertainty(model: nn.Module, x: torch.Tensor, num_samples: int = 100):
    """
    Estimates prediction mean and uncertainty using MC Dropout.

    Args:
        model: the model for prediction.
        x: the input for model.
        num_samples: number of samples.

    Returns:
        mean: Predictive mean.
        std: Predictive standard deviation.
    """
    was_training = model.training
    model.eval()

    for module in model.modules():
        # TODO: implemenet base class so we can just check ifinstance(module, BaseClass)
        if isinstance(module, BayesianLinear) or isinstance(module, BayesianConv2d):
            module.train()

    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(num_samples)])

    mean = preds.mean(dim=0)
    std = preds.std(dim=0, unbiased=True)

    model.train(was_training)

    return mean, std
