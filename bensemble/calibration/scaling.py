"""Post-hoc calibration for classifiers.

Both scalers recalibrate class logits against integer class labels; neither
has a regression counterpart.
"""

import torch
import torch.nn.functional as F
from torch import nn, optim


def _fit_calibrator(
    module: nn.Module,
    params: list[nn.Parameter],
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int,
    optimizer: optim.Optimizer | None,
) -> None:
    """Fits a calibration module by minimizing the negative log-likelihood.

    Args:
        module: Calibration module mapping logits to calibrated logits.
        params: Parameters to optimize, used to build the default optimizer.
        logits: Unscaled logits of shape (N, num_classes).
        labels: Ground truth class indices of shape (N,).
        max_iter: Maximum number of optimizer iterations.
        optimizer: Optimizer over `params`, or None to use the default L-BFGS.
    """
    logits = logits.detach()

    if optimizer is None:
        optimizer = optim.LBFGS(
            params,
            lr=1.0,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

    def eval_loss():
        optimizer.zero_grad()
        loss = F.cross_entropy(module(logits), labels)
        loss.backward()
        return loss

    if isinstance(optimizer, optim.LBFGS):
        optimizer.step(eval_loss)
    else:
        for _ in range(max_iter):
            optimizer.step(eval_loss)


class TemperatureScaling(nn.Module):
    """Temperature Scaling for classifier calibration.

    Divides logits by a single learnable scalar parameter T (temperature).
    This softens probabilities and calibrates confidence without changing
    classification accuracy (argmax remains identical).
    """

    def __init__(self, init_temp: float = 1.5):
        """Initializes the TemperatureScaling module.

        Args:
            init_temp: Initial value for the temperature scalar. Defaults to 1.5.
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies temperature scaling to the input logits.

        Args:
            logits: Raw uncalibrated logits of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Scaled logits of shape (batch_size, num_classes).
        """
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
        optimizer: optim.Optimizer | None = None,
    ) -> "TemperatureScaling":
        """Finds the optimal temperature T using a validation set.

        Optimizes the negative log-likelihood.

        Args:
            logits: Unscaled logits from a hold-out validation set of shape
                (N, num_classes).
            labels: Ground truth class indices of shape (N,).
            max_iter: Maximum number of optimizer iterations. Defaults to 50.
            optimizer: Optimizer over this module's parameters. Defaults to
                L-BFGS with a strong Wolfe line search. L-BFGS runs up to
                `max_iter` iterations within a single step, while any other
                optimizer is stepped `max_iter` times.

        Returns:
            TemperatureScaling: The fitted instance itself.
        """
        _fit_calibrator(self, [self.temperature], logits, labels, max_iter, optimizer)
        return self


class VectorScaling(nn.Module):
    """Vector Scaling for multi-class calibration (extension of Platt Scaling).

    Applies a per-class affine transformation to uncalibrated logits:
    calibrated_logits = logits * a + b
    """

    def __init__(self, num_classes: int):
        """Initializes the VectorScaling module.

        Args:
            num_classes: Number of classes in the classification task.
        """
        super().__init__()
        self.a = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies learned affine transformation to the input logits.

        Args:
            logits: Raw uncalibrated logits of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Calibrated logits of shape (batch_size, num_classes).
        """
        return logits * self.a + self.b

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
        optimizer: optim.Optimizer | None = None,
    ) -> "VectorScaling":
        """Finds optimal scaling vectors 'a' and 'b' using a validation set.

        Optimizes the negative log-likelihood.

        Args:
            logits: Unscaled logits from a hold-out validation set of shape
                (N, num_classes).
            labels: Ground truth class indices of shape (N,).
            max_iter: Maximum number of optimizer iterations. Defaults to 50.
            optimizer: Optimizer over this module's parameters. Defaults to
                L-BFGS with a strong Wolfe line search. L-BFGS runs up to
                `max_iter` iterations within a single step, while any other
                optimizer is stepped `max_iter` times.

        Returns:
            VectorScaling: The fitted instance itself.
        """
        _fit_calibrator(self, [self.a, self.b], logits, labels, max_iter, optimizer)
        return self
