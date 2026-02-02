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


def compute_uncertainty(predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculation of epistemic and aleatory uncertainty"""
    epistemic_uncertainty = predictions.var(dim=0)

    if predictions.dim() > 2:
        mean_probs = predictions.mean(dim=0)
        aleatoric_uncertainty = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
    else:
        aleatoric_uncertainty = torch.zeros_like(epistemic_uncertainty)

    return epistemic_uncertainty, aleatoric_uncertainty


def enable_dropout(model: nn.Module):
    """Activating dropout layers for MC Dropout"""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


class EarlyStopping:
    """Early stop for training"""

    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


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
