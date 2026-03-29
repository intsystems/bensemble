from typing import List, Callable
import torch.nn as nn
from torch.utils.data import DataLoader

from bensemble.core.ensemble import Ensemble


class NeuralEnsembleSearcher:
    def __init__(self):
        pass

    def _forward_select(
        self, pool: List[nn.Module], val_loader: DataLoader
    ) -> List[nn.Module]:
        """
        Args:
            pool: List of trained models.
            val_loader: DataLoader for the validation set.

        Returns:
            List of selected models that minimize the ensemble loss on val_loader.
        """
        pass

    def search_random(
        self,
        train_fn: Callable[[nn.Module], None],
        val_loader: DataLoader,
    ) -> Ensemble:
        """
        Args:
            train_fn (Callable): User-provided function that takes a raw nn.Module and trains it.
            val_loader (DataLoader): Used for the ForwardSelect phase.

        Returns:
            Ensemble: The final ensemble wrapped in bensemble's core abstraction.
        """
        pass
        # models = [model1, model2, ...]
        # return Ensemble.from_models(models)

    def search_evolutionary(
        self,
        train_fn: Callable[[nn.Module], None],
        val_loader: DataLoader,
    ) -> Ensemble:
        """
        Args:
            train_fn (Callable): User-provided function that takes a raw nn.Module and trains it.
            val_loader (DataLoader): Used for the ForwardSelect phase.

        Returns:
            Ensemble: The final ensemble wrapped in bensemble's core abstraction.
        """
        pass
        # models = [model1, model2, ...]
        # return Ensemble.from_models(models)
