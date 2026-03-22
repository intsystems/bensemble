import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for model calibration.

    Divides the logits by a learnable scalar parameter T (temperature).
    This softens the probabilities and calibrates the model's confidence
    without changing its accuracy (the argmax remains the same).
    """

    def __init__(self, init_temp: float = 1.5):
        """
        Args:
            init_temp (float, optional): Initial value for temperature.
                Modern deep networks are typically overconfident, so starting
                with T > 1.0 is recommended. Defaults to 1.5.
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Applies temperature scaling to the input logits.

        Args:
            logits (torch.Tensor): Raw uncalibrated logits of shape[Batch, Num_classes].

        Returns:
            torch.Tensor: Scaled logits of the same shape.
        """
        return logits / self.temperature

    def fit(
        self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50
    ) -> "TemperatureScaling":
        """
        Finds the optimal temperature $T$ using a validation set.

        Uses the L-BFGS optimizer, which is the standard and most efficient
        algorithm for this 1D convex optimization problem.

        Args:
            logits (torch.Tensor): Unscaled logits from a hold-out validation set. Shape: [N, Num_classes].
            labels (torch.Tensor): Ground truth class indices. Shape: [N].
            max_iter (int, optional): Maximum number of L-BFGS iterations. Defaults to 50.

        Returns:
            TemperatureScaling: The fitted model itself.
        """

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        return self
