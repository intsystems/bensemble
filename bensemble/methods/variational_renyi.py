import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base import BaseBayesianEnsemble


class VariationalRenyi(BaseBayesianEnsemble):
    """
    Variational Rényi Bound (VR) Bayesian method.
    Поддерживает регрессию и классификацию.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 1.0,
        prior_sigma: float = 1.0,
        initial_rho: float = -2.0,
        **kwargs
    ):
        """
        Args:
            model: детерминированная PyTorch модель
            alpha: параметр Rényi (α=1 → стандартный VI)
            prior_sigma: σ априорного распределения весов
            initial_rho: инициализация rho для вариационных весов
        """
        # сохраняем шаблон модели для сэмплирования
        self.model_template = copy.deepcopy(model)

        self.alpha = alpha
        self.prior_sigma = prior_sigma
        self.initial_rho = initial_rho

        # bayesian_model: deep-copy и преобразуем слои
        bayesian_model = self._make_bayesian(copy.deepcopy(model))
        self.bayesian_model = bayesian_model

        super().__init__(bayesian_model, **kwargs)
        self.optimizer = None

    def _make_bayesian(self, model: nn.Module) -> nn.Module:
        """Добавляем mu/rho для всех линейных слоёв"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight_mu = nn.Parameter(module.weight.data.clone())
                module.weight_rho = nn.Parameter(
                    torch.full_like(module.weight.data, self.initial_rho)
                )
                if module.bias is not None:
                    module.bias_mu = nn.Parameter(module.bias.data.clone())
                    module.bias_rho = nn.Parameter(
                        torch.full_like(module.bias.data, self.initial_rho)
                    )
                # удаляем старые параметры, чтобы не мешали
                del module._parameters["weight"]
                if module.bias is not None:
                    del module._parameters["bias"]
        return model

    def _reparameterize(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Softplus reparametrization для sigma = softplus(rho)"""
        sigma = torch.log1p(torch.exp(rho))
        sigma = torch.clamp(sigma, min=0.01, max=1.0)  # верхняя граница увеличена
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход с репараметризацией"""
        for module in self.bayesian_model.modules():
            if hasattr(module, "weight_mu"):
                module.weight = self._reparameterize(
                    module.weight_mu, module.weight_rho
                )
                if hasattr(module, "bias_mu") and module.bias_mu is not None:
                    module.bias = self._reparameterize(module.bias_mu, module.bias_rho)
        return self.bayesian_model(x)

    def _compute_log_weights(
        self, batch_X: torch.Tensor, batch_y: torch.Tensor
    ) -> torch.Tensor:
        """Вычисление log_weight для VR bound"""
        output = self.forward(batch_X)
        if output.dim() == 1 or output.shape[1] == 1:  # регрессия
            likelihood_sigma = 0.1  # подбирается под шум данных
            log_likelihood = torch.distributions.Normal(
                output.squeeze(), likelihood_sigma
            ).log_prob(batch_y.squeeze())
        else:  # классификация
            log_likelihood = -F.cross_entropy(output, batch_y, reduction="none")

        # log prior
        log_prior = 0.0
        for module in self.bayesian_model.modules():
            if hasattr(module, "weight_mu"):
                log_prior += (
                    torch.distributions.Normal(0, self.prior_sigma)
                    .log_prob(module.weight)
                    .sum()
                )
                if hasattr(module, "bias_mu") and module.bias_mu is not None:
                    log_prior += (
                        torch.distributions.Normal(0, self.prior_sigma)
                        .log_prob(module.bias)
                        .sum()
                    )

        # log variational
        log_variational = 0.0
        for module in self.bayesian_model.modules():
            if hasattr(module, "weight_mu"):
                sigma_w = torch.log1p(torch.exp(module.weight_rho))
                log_variational += (
                    torch.distributions.Normal(module.weight_mu, sigma_w)
                    .log_prob(module.weight)
                    .sum()
                )
                if hasattr(module, "bias_mu") and module.bias_mu is not None:
                    sigma_b = torch.log1p(torch.exp(module.bias_rho))
                    log_variational += (
                        torch.distributions.Normal(module.bias_mu, sigma_b)
                        .log_prob(module.bias)
                        .sum()
                    )

        return log_likelihood.sum() + log_prior - log_variational

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 100,
        lr: float = 5e-4,
        n_samples: int = 10,
        grad_clip: Optional[float] = 5.0,
        **kwargs
    ) -> Dict[str, List[float]]:

        self.optimizer = torch.optim.Adam(self.bayesian_model.parameters(), lr=lr)
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(num_epochs):
            self.bayesian_model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                log_weights = []
                for _ in range(n_samples):
                    log_w = self._compute_log_weights(batch_X, batch_y)
                    log_weights.append(log_w.unsqueeze(-1))
                log_weights = torch.cat(log_weights, dim=-1)

                # VR bound
                if self.alpha == 1.0:
                    loss = -log_weights.mean()
                else:
                    loss = -(1 / (1 - self.alpha)) * torch.logsumexp(
                        (1 - self.alpha) * log_weights, dim=-1
                    )
                    loss += torch.log(torch.tensor(n_samples, dtype=torch.float))
                    loss = loss.mean()

                loss.backward()

                # gradient clipping
                if grad_clip is not None:
                    for p in self.bayesian_model.parameters():
                        if p.grad is not None:
                            p.grad.data.clamp_(-grad_clip, grad_clip)
                self.optimizer.step()
                train_loss += loss.item()

            history["train_loss"].append(train_loss / len(train_loader))

            if val_loader is not None:
                val_loss = self._validate(val_loader, n_samples)
                history["val_loss"].append(val_loss)

        self.is_fitted = True
        return history

    def _validate(
        self, val_loader: torch.utils.data.DataLoader, n_samples: int
    ) -> float:
        self.bayesian_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                log_weights = []
                for _ in range(n_samples):
                    log_w = self._compute_log_weights(batch_X, batch_y)
                    log_weights.append(log_w.unsqueeze(-1))
                log_weights = torch.cat(log_weights, dim=-1)

                if self.alpha == 1.0:
                    loss = -log_weights.mean()
                else:
                    loss = -(1 / (1 - self.alpha)) * torch.logsumexp(
                        (1 - self.alpha) * log_weights, dim=-1
                    )
                    loss += torch.log(torch.tensor(n_samples, dtype=torch.float))
                    loss = loss.mean()
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def predict(
        self, X: torch.Tensor, n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.bayesian_model.eval()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                predictions.append(self.forward(X).unsqueeze(0))
        predictions = torch.cat(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        return mean_pred, predictions

    def sample_models(self, n_models: int = 10) -> List[nn.Module]:
        models = []
        for _ in range(n_models):
            models.append(self._sample_single_model())
        return models

    def _sample_single_model(self) -> nn.Module:
        model_copy = copy.deepcopy(self.model_template)
        model_copy = self._make_bayesian(model_copy)

        with torch.no_grad():
            for orig, copy_module in zip(
                self.bayesian_model.modules(), model_copy.modules()
            ):
                if hasattr(orig, "weight_mu"):
                    w_sample = self._reparameterize(orig.weight_mu, orig.weight_rho)
                    setattr(copy_module, "weight", w_sample)

                    if hasattr(orig, "bias_mu") and orig.bias_mu is not None:
                        b_sample = self._reparameterize(orig.bias_mu, orig.bias_rho)
                        setattr(copy_module, "bias", b_sample)

        return model_copy

    def _get_ensemble_state(self) -> Dict[str, Any]:
        return {
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "alpha": self.alpha,
        }

    def _set_ensemble_state(self, state: Dict[str, Any]):
        if self.optimizer and state["optimizer_state"]:
            self.optimizer.load_state_dict(state["optimizer_state"])
        self.alpha = state.get("alpha", 1.0)
