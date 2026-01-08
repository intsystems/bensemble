from typing import Tuple

import torch
import torch.nn as nn
import pytorch.lightning as pl


class BensembleModule(pl.LightningModule):
    """
    Abstract base class for bensemble methods.
    """

    def __init__(self, model: nn.Module, *args, **kwargs):
        super().__init__()
        self.model = model

    def predict_step(
        self, batch: any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self.predict(X)

    def predict(
        self, X: torch.Tensor, n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Must be implemented in child class.")
