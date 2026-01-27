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
    """Вычисление эпистемической и алеаторной неопределенности"""
    """Calculation of epistemic and aleatory uncertainty"""
    epistemic_uncertainty = predictions.var(dim=0)

    if predictions.dim() > 2:
        mean_probs = predictions.mean(dim=0)
        aleatoric_uncertainty = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
    else:
        aleatoric_uncertainty = torch.zeros_like(epistemic_uncertainty)

    return epistemic_uncertainty, aleatoric_uncertainty


def enable_dropout(model: nn.Module):
    """Активация dropout слоев для MC Dropout"""
    """Activating dropout layers for MC Dropout"""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


class EarlyStopping:
    """Ранняя остановка для обучения"""

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
