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


class VariationalEnsemble(BaseBayesianEnsemble):
    """
    Variational Inference Ensemble wrapper.

    Converts a standard PyTorch model into a Bayesian Neural Network by replacing
    Linear layers with BayesianLinear layers and optimizing using the ELBO objective.
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood: Optional[nn.Module] = None,
        learning_rate: float = 1e-3,
        prior_sigma: float = 1.0,
        auto_convert: bool = True,
        **kwargs,
    ):
        """
        Args:
            model: The base PyTorch model.
            likelihood: Likelihood module (default: GaussianLikelihood).
            learning_rate: Learning rate for the optimizer.
            prior_sigma: Prior standard deviation for Bayesian layers.
            auto_convert: If True, automatically converts Linear layers to BayesianLinear.
        """
        self.prior_sigma = prior_sigma
        self.learning_rate = learning_rate
        self.likelihood = likelihood if likelihood else GaussianLikelihood()
        self.optimizer = None

        if auto_convert:
            model = self._make_bayesian(model)

        super().__init__(model, **kwargs)

    def _set_sampling_mode(self, active: bool):
        """Toggles the sampling flag in all BayesianLinear layers."""
        for module in self.model.modules():
            if isinstance(module, BayesianLinear):
                module.sampling = active

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Trains the Bayesian model using Evidence Lower Bound (ELBO) maximization.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data (optional).
            epochs: Number of training epochs.
            verbose: If True, prints training progress.
        """
        device = next(self.model.parameters()).device
        if kwargs.get("device"):
            device = torch.device(kwargs["device"])

        self.model.to(device)
        self.likelihood.to(device)

        params = list(self.model.parameters()) + list(self.likelihood.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

        history = {"train_loss": [], "nll": [], "kl": []}

        kl_weight = 1.0 / len(train_loader)

        for epoch in range(epochs):
            self.model.train()
            # Enable sampling for training
            self._set_sampling_mode(True)

            total_loss = 0

            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                self.optimizer.zero_grad()

                # Call without sample=True argument (handled by internal state)
                preds = self.model(X)

                nll = self.likelihood(preds, y)
                kl = self._compute_recursive_kl(self.model)
                loss = nll + kl_weight * kl

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Sigma: {self.likelihood.get_noise_sigma():.4f}"
                )

        self.is_fitted = True
        return history

    def predict(
        self, X: torch.Tensor, n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs Monte Carlo prediction to estimate mean and uncertainty.

        Args:
            X: Input tensor.
            n_samples: Number of MC samples to draw.

        Returns:
            Tuple containing (mean predictions, prediction standard deviation).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        device = next(self.model.parameters()).device
        X = X.to(device)
        self.model.eval()

        self._set_sampling_mode(True)

        sampled_preds = []
        noise_sigma = self.likelihood.get_noise_sigma()

        with torch.no_grad():
            for _ in range(n_samples):
                preds = self.model(X)

                eps = torch.randn_like(preds) * noise_sigma
                sampled_preds.append(preds + eps)

        stack = torch.stack(sampled_preds)
        mean = stack.mean(dim=0)
        std = stack.std(dim=0)
        return mean.cpu(), std.cpu()

    def sample_models(self, n_models: int = 10) -> List[nn.Module]:
        """
        Samples specific deterministic models from the posterior distribution.

        Args:
            n_models: Number of models to sample.

        Returns:
            List of PyTorch models with fixed weights sampled from the posterior.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        models = []
        for _ in range(n_models):
            model_sample = copy.deepcopy(self.model)
            model_sample.eval()

            for module in model_sample.modules():
                if isinstance(module, BayesianLinear):
                    w_sigma = F.softplus(module.w_rho)
                    b_sigma = F.softplus(module.b_rho)

                    # Generate fixed weights
                    w_sample = module.w_mu + w_sigma * torch.randn_like(module.w_mu)
                    b_sample = module.b_mu + b_sigma * torch.randn_like(module.b_mu)

                    module.w_mu.data = w_sample
                    module.b_mu.data = b_sample

                    # Disable sampling within the layer permanently
                    module.sampling = False

            models.append(model_sample)
        return models

    def _compute_recursive_kl(self, module: nn.Module) -> torch.Tensor:
        kl = 0
        for child in module.children():
            if hasattr(child, "kl_divergence"):
                kl += child.kl_divergence()
            else:
                kl += self._compute_recursive_kl(child)
        return kl

    def _make_bayesian(self, model: nn.Module) -> nn.Module:
        """
        Recursively replaces standard Linear layers with BayesianLinear layers.
        """

        def replace_layers(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    new_layer = BayesianLinear(
                        child.in_features, child.out_features, self.prior_sigma
                    )
                    with torch.no_grad():
                        new_layer.w_mu.data.copy_(child.weight.data)
                        if child.bias is not None:
                            new_layer.b_mu.data.copy_(child.bias.data)
                    setattr(module, name, new_layer)
                else:
                    replace_layers(child)

        replace_layers(model)
        return model

    def _get_ensemble_state(self):
        state = {
            "model_state_dict": self.model.state_dict(),
            "likelihood_state_dict": self.likelihood.state_dict(),
            "is_fitted": self.is_fitted,
            "prior_sigma": self.prior_sigma,
            "learning_rate": self.learning_rate,
        }

        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        return state

    def _set_ensemble_state(self, state):
        self.is_fitted = state["is_fitted"]
        self.prior_sigma = state["prior_sigma"]
        self.learning_rate = state["learning_rate"]

        self.model.load_state_dict(state["model_state_dict"])
        self.likelihood.load_state_dict(state["likelihood_state_dict"])

        if self.optimizer is not None and "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
