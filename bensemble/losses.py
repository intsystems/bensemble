import math

import torch
import torch.nn.functional as F
from torch import nn


class GaussianLikelihood(nn.Module):
    """Gaussian Likelihood with learnable homoscedastic uncertainty.

    Learns a global standard deviation (sigma) for data noise and computes
    negative log-likelihood via Gaussian NLL.
    """

    def __init__(self, init_log_sigma: float = -2.0):
        """Initializes the GaussianLikelihood layer.

        Args:
            init_log_sigma: Initial value for log standard deviation. Defaults to -2.0.
        """
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor([init_log_sigma]))
        self.loss_fn = nn.GaussianNLLLoss(reduction="none")

    @property
    def sigma(self) -> float:
        """float: Current standard deviation value."""
        return F.softplus(self.log_sigma).item() + 1e-6

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes Gaussian negative log-likelihood loss.

        Args:
            preds: Predicted tensor of shape (*).
            target: Target tensor of shape (*).

        Returns:
            torch.Tensor: Element-wise negative log-likelihood tensor matching target shape.
        """
        sigma = F.softplus(self.log_sigma) + 1e-6
        var = sigma.pow(2)
        var_expanded = var.expand_as(preds)

        return self.loss_fn(preds, target, var_expanded)


class VariationalLoss(nn.Module):
    """Variational loss supporting ELBO and Rényi alpha-divergence."""

    def __init__(
        self,
        likelihood_model: nn.Module,
        alpha: float = 1.0,
        num_batches: int = 1,
        kl_weight: float = 1.0,
    ):
        """Initializes the VariationalLoss module.

        Args:
            likelihood_model: Module computing negative log-likelihood p(y|x).
            alpha: Rényi alpha parameter. When alpha=1.0, standard ELBO is used. Defaults to 1.0.
            num_batches: Total number of training batches in an epoch (for KL mini-batch scaling). Defaults to 1.
            kl_weight: Additional scaling factor for KL divergence term. Defaults to 1.0.
        """
        super().__init__()
        self.likelihood_model = likelihood_model
        self.alpha = alpha
        self.num_batches = num_batches
        self.kl_weight = kl_weight

    def forward(
        self, preds: torch.Tensor, target: torch.Tensor, kl_divergence: torch.Tensor
    ) -> torch.Tensor:
        """Computes variational objective loss.

        Args:
            preds: Predictions tensor of shape (K, batch_size, ...) or (batch_size, ...).
            target: Target tensor of shape (batch_size, ...).
            kl_divergence: Total KL divergence of model parameters.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if preds.ndim == target.ndim:
            preds = preds.unsqueeze(0)

        k_samples = preds.size(0)

        log_likelihoods = -self.likelihood_model(preds, target)
        log_likelihoods = (
            log_likelihoods.sum(dim=1) if log_likelihoods.ndim > 1 else log_likelihoods
        )

        kl_scaled = (kl_divergence / self.num_batches) * self.kl_weight
        log_weights = log_likelihoods - kl_scaled

        if abs(self.alpha - 1.0) < 1e-6:
            return -(log_weights.mean())
        else:
            term = (1 - self.alpha) * log_weights
            log_sum_exp = torch.logsumexp(term, dim=0)
            loss = -(1 / (1 - self.alpha)) * (log_sum_exp - math.log(k_samples))
            return loss
