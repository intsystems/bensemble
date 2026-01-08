from typing import Tuple
from typing import List
from typing import Optional
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

    @abc.abstractmethod
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[object]]:
        """Returns (optimizer, scheduler). Scheduler can be None."""


class Callback:
    """Base class for callbacks"""

    def on_train_start(self, trainer, model): ...
    def on_epoch_start(self, trainer, model): ...
    def on_batch_start(self, trainer, model): ...
    def on_batch_end(self, trainer, model): ...
    def on_epoch_end(self, trainer, model): ...
    def on_train_end(self, trainer, model): ...


class Trainer:
    """Trainer class for Bensemble Methods."""

    def __init__(
        self,
        max_epochs: int = 100,
        device: Optional[str] = None,
        callbacks: List[Callback] = None,
    ):
        self.max_epochs = max_epochs
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.callbacks = callbacks

        self.optimizer = None
        self.scheduler = None

    def fit(self, model: BayesianEnsemble, train_loader, val_loader=None):
        """
        Trains the given model.
        """
        model.to(self.device)

        self.optimizer, self.scheduler = model.configure_optimizers()
        dataset_size = len(train_loader.dataset)

        for cb in self.callbacks:
            cb.on_train_start(self, model)

        for epoch in range(self.max_epochs):
            for cb in self.callbacks:
                cb.on_epoch_start(self, model, epoch)

            model.train()
            train_metrics = {}
            n_batches = len(train_loader)

            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()

                output = model.training_step(batch, batch_idx, dataset_size)
                loss = output["loss"]

                loss.backward()
                self.optimizer.step()

                for k, v in output.items():
                    val = v.item()
                    train_metrics[k] = train_metrics.get(k, 0) + val

                for cb in self.callbacks:
                    cb.on_batch_end(self, model, output, batch_idx)

            avg_metrics = {
                f"train_{k}": v / n_batches for k, v in train_metrics.items()
            }

            if val_loader:
                model.eval()
                val_metrics = {}
                n_val = len(val_loader)

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        output = model.validation_step(batch, batch_idx)

                        for k, v in output.items():
                            val = v.item()
                            val_metrics[k] = val_metrics.get(k, 0) + val

                for k, v in val_metrics.items():
                    avg_metrics[f"val_{k}"] = v / n_val

            for cb in self.callbacks:
                cb.on_epoch_end(self, model, epoch, avg_metrics)

        for cb in self.callbacks:
            cb.on_train_end(self, model)
