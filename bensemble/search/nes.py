from typing import Callable
from torch.utils.data import DataLoader
import torch.nn as nn

from bensemble.core.ensemble import Ensemble


class RandomSearcher:
    def __init__(self):
        pass

    def search(
        self, train_fn: Callable[[nn.Module], None], val_loader: DataLoader
    ) -> Ensemble:
        """
        Args:
            train_fn (Callable): User-provided function that takes a raw nn.Module and trains it.
            val_loader (DataLoader): Used for the ForwardSelect phase to evaluate ensemble performance.

        Returns:
            Ensemble: The final ensemble wrapped in bensemble's core abstraction.
        """
        pass
        # models = [model1, model2, ...]
        # return Ensemble.from_models(models)


class EvolutionarySearcher:
    def __init__(self):
        pass

    def search(
        self, train_fn: Callable[[nn.Module], None], val_loader: DataLoader
    ) -> Ensemble:
        """
        Args:
            train_fn (Callable): User-provided function that takes a raw nn.Module and trains it.
            val_loader (DataLoader): Used for tournament selection and the final ForwardSelect phase.

        Returns:
            Ensemble: The final ensemble wrapped in bensemble's core abstraction.
        """
        pass
        # models = [model1, model2, ...]
        # return Ensemble.from_models(models)
