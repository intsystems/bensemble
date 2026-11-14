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
            likelihood_model: Module returning element-wise or summed negative
                log-likelihood. Use reduction="none" or "sum", not "mean".
            alpha: Rényi parameter. At 1.0, use the ELBO. Other values require
                sampled log prior and posterior densities in forward().
            num_batches: Number of equal-sized training batches. The Rényi path
                estimates the full-data log likelihood, then divides the final
                objective by this number. Defaults to 1 (full-batch training).
            kl_weight: Weight on the prior/posterior log ratio or analytic KL.
                The standard variational objective uses 1.0.
        """
        super().__init__()
        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite.")
        if (
            isinstance(num_batches, bool)
            or not isinstance(num_batches, int)
            or num_batches < 1
        ):
            raise ValueError("num_batches must be a positive integer.")
        if not math.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("kl_weight must be finite and non-negative.")
        if alpha != 1.0 and getattr(likelihood_model, "reduction", None) == "mean":
            raise ValueError(
                'Rényi loss requires likelihood reduction="none" or "sum".'
            )
        self.likelihood_model = likelihood_model
        self.alpha = alpha
        self.num_batches = num_batches
        self.kl_weight = kl_weight

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        kl_divergence: torch.Tensor | None = None,
        *,
        log_prior: torch.Tensor | None = None,
        log_posterior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes variational objective loss.

        Args:
            preds: Predictions tensor of shape (K, batch_size, ...) or (batch_size, ...).
            target: Target tensor of shape (batch_size, ...).
            kl_divergence: Analytic KL, supported only for alpha=1.0. Omit when
                providing sampled log densities.
            log_prior: Log prior density of each weight draw, shape (K,).
            log_posterior: Log posterior density of the same draws, shape (K,).
                Each pair must describe the weights used for preds[k]. Use
                bensemble.utils.sample_with_log_probs to generate matched draws.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        sampled = log_prior is not None or log_posterior is not None
        if sampled:
            if log_prior is None or log_posterior is None:
                raise ValueError("Provide both log_prior and log_posterior.")
            if kl_divergence is not None:
                raise ValueError(
                    "Provide either kl_divergence or sampled log densities, not both."
                )
        elif self.alpha != 1.0:
            raise ValueError(
                "Rényi loss requires per-sample log_prior and log_posterior; "
                "a scalar KL is not sufficient. Use sample_with_log_probs()."
            )
        elif kl_divergence is None:
            raise ValueError("ELBO requires kl_divergence or sampled log densities.")

        class_indices = not torch.is_floating_point(target)

        if class_indices:
            if preds.ndim == target.ndim + 1:
                preds = preds.unsqueeze(0)
            log_likelihoods = -torch.stack(
                [self.likelihood_model(sample, target) for sample in preds]
            )
        else:
            has_sample_axis = (
                sampled
                and log_prior.shape == (preds.shape[0],)
                and preds.shape[1:] == target.shape
            )
            if preds.ndim == target.ndim:
                preds = preds.unsqueeze(0)
            elif not has_sample_axis and target.shape == preds.shape[:-1]:
                target = target.unsqueeze(-1)
                preds = preds.unsqueeze(0)
            while target.ndim < preds.ndim - 1:
                target = target.unsqueeze(-1)
            log_likelihoods = -torch.stack(
                [self.likelihood_model(sample, target) for sample in preds]
            )

        k_samples = preds.size(0)
        if log_likelihoods.dtype in (torch.float16, torch.bfloat16):
            log_likelihoods = log_likelihoods.float()
        log_likelihoods = log_likelihoods.reshape(k_samples, -1).sum(dim=1)

        if not sampled:
            kl_scaled = (kl_divergence / self.num_batches) * self.kl_weight
            return -(log_likelihoods - kl_scaled).mean()

        for name, values in (
            ("log_prior", log_prior),
            ("log_posterior", log_posterior),
        ):
            if values.shape != (k_samples,) or not torch.is_floating_point(values):
                raise ValueError(
                    f"{name} must be a floating tensor of shape ({k_samples},)."
                )
            if values.device != log_likelihoods.device:
                raise ValueError(f"{name} must be on the same device as preds.")

        log_weights = self.num_batches * log_likelihoods + self.kl_weight * (
            log_prior - log_posterior
        )
        center = log_weights.mean()
        if self.alpha == 1.0:
            return -center / self.num_batches

        scale = 1 - self.alpha
        centered = scale * (log_weights - center)
        # Avoid cancellation in logsumexp(x) - log(K) near alpha=1.
        if centered.detach().abs().max() < 0.5:
            log_mean_exp = torch.log1p(torch.expm1(centered).mean())
        else:
            log_mean_exp = torch.logsumexp(centered, dim=0) - math.log(k_samples)
        return -(center + log_mean_exp / scale) / self.num_batches
