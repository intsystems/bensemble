import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseBayesianLayer(nn.Module):
    """
    Base class for all bayesian layers.

    Computes KL-divergence automatically for all parameters ending with `_mu` and `_rho`.
    """

    def __init__(self, prior_sigma: float = 1.0):
        super().__init__()
        self.prior_sigma = prior_sigma

    def kl_divergence(self) -> torch.Tensor:
        """
        Computes KL-divergence KL(q || p) for all bayesian weights of the layer.
        p(w) = N(0, prior_sigma^2)
        q(w) = N(mu, sigma^2), where sigma = softplus(rho)
        """
        total_kl = 0.0

        for name, param in self.named_parameters():
            if name.endswith("_mu"):
                rho_name = name.replace("_mu", "_rho")

                if hasattr(self, rho_name):
                    mu = param
                    rho = getattr(self, rho_name)

                    total_kl += self._compute_kl_for_param(mu, rho)

        return total_kl

    def _compute_kl_for_param(
        self, mu: torch.Tensor, rho: torch.Tensor
    ) -> torch.Tensor:
        """Computes KL-divergence for pair (mu, rho)."""
        sigma = F.softplus(rho)

        num_el = mu.numel()

        const_term = -0.5 * num_el
        log_prior_term = math.log(self.prior_sigma) * num_el

        kl = -torch.log(sigma).sum() + (sigma**2 + mu**2).sum() / (
            2 * self.prior_sigma**2
        )

        return kl + log_prior_term + const_term
