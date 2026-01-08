import abc
from typing import Any, Dict

import torch
import torch.nn as nn


class BayesianEnsemble(nn.Module, abc.ABC):
    """
    Interface for Bensemble Methods.
    """

    @abc.abstractmethod
    def training_step(
        self, batch: Any, batch_idx: int, dataset_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary. It must have a key 'loss'.
        Other optional keys are metrics for logging.
        """
        pass

    @abc.abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary. It must have a key 'loss'.
        Other optional keys are metrics for logging.
        This method is the same as training_step(), but it does not calculate gradients.
        """
        pass
