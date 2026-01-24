import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Normal, kl_divergence


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
        self.init_sigma = init_sigma
        self.weight_init = weight_init

        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features))

        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_rho = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_init == "kaiming":
            init.kaiming_normal_(self.w_mu, nonlinearity="relu")
        elif self.weight_init == "xavier":
            init.xavier_normal_(self.w_mu)
        else:
            init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))

        init.zeros_(self.b_mu)

        rho_init_val = math.log(math.exp(self.init_sigma) - 1.0)

        self.w_rho.data.fill_(rho_init_val)
        self.b_rho.data.fill_(rho_init_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Local Reparameterization Trick.
        """
        if not self.training:
            return F.linear(x, self.w_mu, self.b_mu)

        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)

        gamma = F.linear(x, self.w_mu)
        delta = F.linear(x.pow(2), w_sigma.pow(2)) + b_sigma.pow(2)

        eps = torch.randn_like(gamma)
        return gamma + eps * torch.sqrt(delta + 1e-8) + self.b_mu

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
