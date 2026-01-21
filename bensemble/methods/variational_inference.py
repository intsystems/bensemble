import copy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

from ..core.base import BaseBayesianEnsemble


class BayesianLinear(nn.Module):
    """
    Bayesian Linear layer implementing Variational Inference with the
    Local Reparameterization Trick.

    Weights and biases are modeled as Gaussian distributions with learnable
    means and standard deviations (parametrized by rho).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        init_sigma: float = 0.1,
        weight_init: str = "kaiming",
    ):
        """
        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            prior_sigma: Standard deviation of the prior Gaussian distribution.
            init_sigma: Initial standard deviation for the posterior.
            weight_init: Initialization method for weight means ('kaiming', 'xavier', or 'normal').
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Mode flag: True = sample noise (training/MC), False = use mean (deterministic)
        self.sampling = True

        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features))

        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.ones(out_features))

        if weight_init == "kaiming":
            init.kaiming_normal_(self.w_mu, nonlinearity="linear")
        elif weight_init == "xavier":
            init.xavier_normal_(self.w_mu)
        else:
            init.normal_(self.w_mu, mean=0.0, std=0.05)

        init.zeros_(self.b_mu)

        def inverse_softplus(s):
            s = float(s)
            if s <= 1e-6:
                return math.log(math.exp(1e-6) - 1.0)
            return math.log(math.exp(s) - 1.0)

        rho_init = inverse_softplus(init_sigma)
        self.w_rho.data.fill_(rho_init)
        self.b_rho.data.fill_(rho_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Local Reparameterization Trick.
        """
        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)

        gamma = F.linear(x, self.w_mu)

        if self.sampling:
            # Local Reparameterization Trick
            delta = F.linear(x.pow(2), w_sigma.pow(2)) + b_sigma.pow(2)
            eps = torch.randn_like(gamma)
            out = gamma + eps * torch.sqrt(delta + 1e-8)
            return out + self.b_mu
        else:
            return gamma + self.b_mu

    def kl_divergence(self) -> torch.Tensor:
        """
        Computes the KL divergence between the variational posterior and the prior.
        """
        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)

        q_w = Normal(self.w_mu, w_sigma)
        p_w = Normal(
            torch.zeros_like(self.w_mu), torch.full_like(w_sigma, self.prior_sigma)
        )

        q_b = Normal(self.b_mu, b_sigma)
        p_b = Normal(
            torch.zeros_like(self.b_mu), torch.full_like(b_sigma, self.prior_sigma)
        )

        kl_w = kl_divergence(q_w, p_w).sum()
        kl_b = kl_divergence(q_b, p_b).sum()
        return kl_w + kl_b


class GaussianLikelihood(nn.Module):
    """
    Gaussian Likelihood with a learnable noise parameter (homoscedastic aleatoric uncertainty).
    """

    def __init__(self, init_log_sigma: float = -2.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor([init_log_sigma]))
        self.loss_fn = nn.GaussianNLLLoss(reduction="sum", full=False)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Negative Log Likelihood loss.
        """
        sigma = F.softplus(self.log_sigma) + 1e-3
        var = (sigma**2).expand_as(preds)
        return self.loss_fn(preds, target, var)

    def get_noise_sigma(self) -> float:
        """
        Returns the current estimated noise standard deviation.
        """
        return (F.softplus(self.log_sigma) + 1e-3).item()


# TODO: we should remove this, the user is going to build its own models using basic "bricks"
class VariationalEnsemble(BaseBayesianEnsemble):
    """
    Variational Inference Ensemble wrapper.

    Converts a standard PyTorch model into a Bayesian Neural Network by replacing
    Linear layers with BayesianLinear layers and optimizing using the ELBO objective.
    """

    pass
