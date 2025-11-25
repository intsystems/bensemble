import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from typing import List, Optional, Tuple

from ..core.base import BaseBayesianEnsemble


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Флаг режима: True = сэмплируем шум (обучение/MC), False = используем среднее
        self.sampling = True

        # Инициализация параметров
        self.w_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.w_rho = nn.Parameter(torch.ones(out_features, in_features) * -3.0)

        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.ones(out_features) * -3.0)

    def forward(self, x):
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

    def kl_divergence(self):
        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)

        def kl_term(mu, sigma):
            var_q = sigma.pow(2)
            var_p = self.prior_sigma**2
            return (
                torch.log(self.prior_sigma / sigma)
                + (var_q + mu.pow(2)) / (2 * var_p)
                - 0.5
            ).sum()

        return kl_term(self.w_mu, w_sigma) + kl_term(self.b_mu, b_sigma)


class GaussianLikelihood(nn.Module):
    def __init__(self, init_log_sigma=-2.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor([init_log_sigma]))

    def forward(self, preds, target):
        sigma = F.softplus(self.log_sigma) + 1e-3
        mse_term = 0.5 * (target - preds) ** 2 / sigma**2
        return (mse_term + torch.log(sigma)).sum()

    def get_noise_sigma(self):
        return (F.softplus(self.log_sigma) + 1e-3).item()


class VariationalEnsemble(BaseBayesianEnsemble):
    def __init__(
        self,
        model: nn.Module,
        likelihood: Optional[nn.Module] = None,
        learning_rate: float = 1e-3,
        prior_sigma: float = 1.0,
        auto_convert: bool = True,
        **kwargs,
    ):
        self.prior_sigma = prior_sigma
        self.learning_rate = learning_rate
        self.likelihood = likelihood if likelihood else GaussianLikelihood()
        self.optimizer = None

        if auto_convert:
            model = self._make_bayesian(model)

        super().__init__(model, **kwargs)

    def _set_sampling_mode(self, active: bool):
        """Переключает флаг sampling во всех слоях BayesianLinear"""
        for module in self.model.modules():
            if isinstance(module, BayesianLinear):
                module.sampling = active

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=100,
        kl_weight=0.1,
        verbose=True,
        **kwargs,
    ):
        device = next(self.model.parameters()).device
        if kwargs.get("device"):
            device = torch.device(kwargs["device"])

        self.model.to(device)
        self.likelihood.to(device)

        params = list(self.model.parameters()) + list(self.likelihood.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

        history = {"train_loss": [], "nll": [], "kl": []}

        for epoch in range(epochs):
            self.model.train()
            # включаем сэмплирование для обучения
            self._set_sampling_mode(True)

            total_loss = 0

            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                self.optimizer.zero_grad()

                # Вызов теперь без аргумента sample=True
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

                    # Генерируем фиксированные веса
                    w_sample = module.w_mu + w_sigma * torch.randn_like(module.w_mu)
                    b_sample = module.b_mu + b_sigma * torch.randn_like(module.b_mu)

                    module.w_mu.data = w_sample
                    module.b_mu.data = b_sample

                    # Отключаем сэмплирование внутри слоя навсегда
                    module.sampling = False

            models.append(model_sample)
        return models

    def _compute_recursive_kl(self, module):
        kl = 0
        for child in module.children():
            if hasattr(child, "kl_divergence"):
                kl += child.kl_divergence()
            else:
                kl += self._compute_recursive_kl(child)
        return kl

    def _make_bayesian(self, model: nn.Module) -> nn.Module:
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

    # Временные заглушки!!!!!
    def _get_ensemble_state(self):
        return {}

    def _set_ensemble_state(self, state):
        pass
