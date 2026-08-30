import torch
import torch.nn.functional as F
from torch import nn, optim


class TemperatureScaling(nn.Module):
    """Temperature Scaling for model calibration.

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
        self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50
    ) -> "TemperatureScaling":
        """Finds the optimal temperature T using a validation set.

        Optimizes the negative log-likelihood via L-BFGS.

        Args:
            logits: Unscaled logits from a hold-out validation set of shape
                (N, num_classes).
            labels: Ground truth class indices of shape (N,).
            max_iter: Maximum number of L-BFGS iterations. Defaults to 50.

        Returns:
            TemperatureScaling: The fitted instance itself.
        """
        logits = logits.detach()
        optimizer = optim.LBFGS(
            [self.temperature],
            lr=1.0,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
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
        self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50
    ) -> "VectorScaling":
        """Finds optimal scaling vectors 'a' and 'b' using a validation set.

        Optimizes the negative log-likelihood via L-BFGS.

        Args:
            logits: Unscaled logits from a hold-out validation set of shape
                (N, num_classes).
            labels: Ground truth class indices of shape (N,).
            max_iter: Maximum number of L-BFGS iterations. Defaults to 50.

        Returns:
            VectorScaling: The fitted instance itself.
        """
        logits = logits.detach()
        optimizer = optim.LBFGS(
            [self.a, self.b],
            lr=1.0,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self
