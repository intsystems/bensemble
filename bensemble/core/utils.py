from typing import Tuple

import torch
import torch.nn as nn


def compute_uncertainty(predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Вычисление эпистемической и алеаторной неопределенности
    """
    # Эпистемическая неопределенность (вариация между моделями)
    epistemic_uncertainty = predictions.var(dim=0)

    # Алеаторная неопределенность (средняя энтропия предсказаний)
    if predictions.dim() > 2:  # Классификация
        mean_probs = predictions.mean(dim=0)
        aleatoric_uncertainty = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
    else:  # Регрессия
        aleatoric_uncertainty = torch.zeros_like(epistemic_uncertainty)

    return epistemic_uncertainty, aleatoric_uncertainty


def enable_dropout(model: nn.Module):
    """Активация dropout слоев для MC Dropout"""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


class EarlyStopping:
    """Ранняя остановка для обучения"""

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
