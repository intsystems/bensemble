import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianLikelihood(nn.Module):
    """
    Gaussian Likelihood with learnable homoscedastic uncertainty.

    This layer learns a single global standard deviation (sigma) for the data noise.
    It wraps `nn.GaussianNLLLoss` to compute the negative log likelihood.
    """

    def __init__(self, init_log_sigma: float = -2.0):
        """
        Args:
            init_log_sigma (float): Initial value for the log of standard deviation.
                                    Smaller values mean less initial noise assumption.
        """

        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor([init_log_sigma]))
        self.loss_fn = nn.GaussianNLLLoss(reduction="none")

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(self.log_sigma) + 1e-6
        var = sigma.pow(2)
        var_expanded = var.expand_as(preds)

        return self.loss_fn(preds, target, var_expanded)


class VariationalLoss(nn.Module):
    def __init__(
        self, likelihood_model: nn.Module, alpha: float = 1.0, num_batches: int = 1
    ):
        """
        Args:
            likelihood_model: Module for likelihood calculation p(y|x)
            alpha: Rényi Parameter. 1.0 = ELBO.
            num_batches: Num of batches in a epoch.
        """
        super().__init__()
        self.likelihood_model = likelihood_model
        self.alpha = alpha
        self.num_batches = num_batches

    def forward(
        self, preds: torch.Tensor, target: torch.Tensor, kl_divergence: torch.Tensor
    ) -> torch.Tensor:
        if preds.ndim == target.ndim:
            preds = preds.unsqueeze(0)

        k_samples = preds.size(0)

        log_likelihoods = -self.likelihood_model(preds, target)
        log_likelihoods = (
            log_likelihoods.sum(dim=1) if log_likelihoods.ndim > 1 else log_likelihoods
        )

        kl_per_batch = kl_divergence / self.num_batches
        log_weights = log_likelihoods - kl_per_batch

        if abs(self.alpha - 1.0) < 1e-6:
            # ELBO case
            return -(log_weights.mean())
        else:
            term = (1 - self.alpha) * log_weights
            log_sum_exp = torch.logsumexp(term, dim=0)
            loss = -(1 / (1 - self.alpha)) * (log_sum_exp - math.log(k_samples))
            return loss
