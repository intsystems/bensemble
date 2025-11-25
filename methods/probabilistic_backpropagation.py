import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..core.base import BaseBayesianEnsemble


def phi(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def relu_moments(m: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    v = torch.clamp(v, min=eps)
    sigma = torch.sqrt(v)
    alpha = m / sigma
    alpha_eval = torch.clamp(alpha, min=-10.0, max=10.0)
    pdf = phi(alpha_eval)
    cdf = Phi(alpha_eval)
    mean = sigma * pdf + m * cdf
    second_moment = (v + m * m) * cdf + m * sigma * pdf
    var = torch.clamp(second_moment - mean * mean, min=eps)
    return mean, var


class ProbLinear(nn.Module):
    """Линейный слой с параметрами среднего и дисперсии для PBP."""

    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        d = self.in_features + 1  # +1 для bias
        h = self.out_features
        scale = 1.0 / math.sqrt(d)
        self.m = nn.Parameter(scale * torch.randn(h, d, dtype=self.dtype, device=self.device))
        self.v = nn.Parameter(0.5 * torch.ones(h, d, dtype=self.dtype, device=self.device))


class PBPNet(nn.Module):
    """Сеть на основе ProbLinear с аналитическим распространением моментов."""

    def __init__(self, layer_sizes: List[int], dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.layers: List[ProbLinear] = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(ProbLinear(layer_sizes[i], layer_sizes[i + 1], dtype=dtype, device=device))
        self.dtype = dtype
        self.device = device or torch.device("cpu")

    def forward_moments(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(device=self.device, dtype=self.dtype)
        assert x.dim() == 2 and x.shape[0] >= 1
        batch = x.shape[0]

        mz = torch.cat([x, torch.ones(batch, 1, device=self.device, dtype=self.dtype)], dim=1)  # (B, D+1)
        vz = torch.zeros_like(mz)

        for li, layer in enumerate(self.layers):
            d = layer.in_features + 1
            scale = 1.0 / math.sqrt(d)

            ma = (mz @ layer.m.t()) * scale  # (B, H)
            term1 = (vz @ (layer.m ** 2).t())
            term2 = ((mz ** 2) @ layer.v.t())
            term3 = (vz @ layer.v.t())
            va = (term1 + term2 + term3) * (scale ** 2)

            is_last = (li == len(self.layers) - 1)
            if not is_last:
                mb, vb = relu_moments(ma, va)
                mz = torch.cat([mb, torch.ones(batch, 1, device=self.device, dtype=self.dtype)], dim=1)
                vz = torch.cat([vb, torch.zeros(batch, 1, device=self.device, dtype=self.dtype)], dim=1)
            else:
                mz = ma
                vz = va

        return mz, vz


class ProbabilisticBackpropagation(BaseBayesianEnsemble):
    """
    Probabilistic Backpropagation (PBP) for Bayesian regression with moment matching.
    Основано на демо ноутбуке pbp_demo.ipynb.
    """

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 layer_sizes: Optional[List[int]] = None,
                 noise_alpha: float = 6.0,
                 noise_beta: float = 6.0,
                 weight_alpha: float = 6.0,
                 weight_beta: float = 6.0,
                 dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None):
        if model is None:
            if layer_sizes is None:
                raise ValueError("Specify either a ready PBP model or layer_sizes to construct it.")
            model = PBPNet(layer_sizes, dtype=dtype, device=device)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        super().__init__(model.to(self.device))

        self.alpha_g = torch.tensor(noise_alpha, dtype=self.dtype, device=self.device)
        self.beta_g = torch.tensor(noise_beta, dtype=self.dtype, device=self.device)
        self.alpha_l = torch.tensor(weight_alpha, dtype=self.dtype, device=self.device)
        self.beta_l = torch.tensor(weight_beta, dtype=self.dtype, device=self.device)

    def _logZ_gaussian_likelihood(self,
                                  y: torch.Tensor,
                                  mz: torch.Tensor,
                                  vz: torch.Tensor,
                                  alpha_g: torch.Tensor,
                                  beta_g: torch.Tensor,
                                  eps: float = 1e-12) -> torch.Tensor:
        alpha_g = torch.clamp(alpha_g, min=1.0 + 1e-6)
        sigma2_eff = beta_g / (alpha_g - 1.0) + vz
        sigma2_eff = torch.clamp(sigma2_eff, min=eps)
        diff = y.reshape_as(mz) - mz
        return (-0.5 * (diff * diff) / sigma2_eff - 0.5 * torch.log(2.0 * math.pi * sigma2_eff)).squeeze(-1)

    def _gamma_adf_update_from_Z(self,
                                 logZ: torch.Tensor,
                                 logZ1: torch.Tensor,
                                 logZ2: torch.Tensor,
                                 alpha_old: torch.Tensor,
                                 beta_old: torch.Tensor,
                                 clamp_eps: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
        r1 = torch.exp(logZ1 - logZ)
        r2 = torch.exp(logZ2 - logZ)

        term_alpha = (r2 / (r1 * r1)) * ((alpha_old + 1.0) / alpha_old)
        denom_alpha = torch.clamp(term_alpha - 1.0, min=1e-12)
        alpha_new = 1.0 / denom_alpha

        termA = r2 / r1 * ((alpha_old + 1.0) / beta_old)
        termB = r1 * (alpha_old / beta_old)
        denom_beta = torch.clamp(termA - termB, min=clamp_eps)
        beta_new = 1.0 / denom_beta

        alpha_new = torch.clamp(alpha_new, min=1.0 + 1e-6, max=1e6)
        beta_new = torch.clamp(beta_new, min=clamp_eps, max=1e9)
        return alpha_new.detach(), beta_new.detach()

    def _single_datapoint_adf_step(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   step_clip: Optional[float] = 1.0) -> None:
        for layer in self.model.layers:
            layer.m.requires_grad_(True)
            layer.v.requires_grad_(True)

        mz, vz = self.model.forward_moments(x.reshape(1, -1))
        logZ = self._logZ_gaussian_likelihood(y.reshape(1, -1), mz, vz, self.alpha_g, self.beta_g).mean()
        logZ1 = self._logZ_gaussian_likelihood(y.reshape(1, -1), mz, vz, self.alpha_g + 1.0, self.beta_g).mean()
        logZ2 = self._logZ_gaussian_likelihood(y.reshape(1, -1), mz, vz, self.alpha_g + 2.0, self.beta_g).mean()

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        logZ.backward()

        for layer in self.model.layers:
            gm = layer.m.grad
            gv = layer.v.grad

            if step_clip is not None:
                gm = torch.clamp(gm, min=-step_clip, max=step_clip)
                gv = torch.clamp(gv, min=-step_clip, max=step_clip)

            m = layer.m
            v = torch.clamp(layer.v, min=1e-10, max=1e3)

            m_new = m + v * gm
            v_new = v - (v * v) * (gm * gm - 2.0 * gv)
            v_new = torch.clamp(v_new, min=1e-10, max=1e3)

            with torch.no_grad():
                layer.m.copy_(m_new.detach())
                layer.v.copy_(v_new.detach())

        self.alpha_g, self.beta_g = self._gamma_adf_update_from_Z(
            logZ.detach(), logZ1.detach(), logZ2.detach(), self.alpha_g, self.beta_g
        )

        for layer in self.model.layers:
            layer.m.grad = None
            layer.v.grad = None

    def _prior_refresh_epoch(self, n_refresh: int = 1) -> None:
        alpha_l = torch.clamp(self.alpha_l, min=1.0 + 1e-6)
        beta_l = self.beta_l
        for _ in range(n_refresh):
            s2 = beta_l / (alpha_l - 1.0)
            s2_1 = beta_l / (alpha_l)
            s2_2 = beta_l / (alpha_l + 1.0)

            Z_acc = 0.0
            Z1_acc = 0.0
            Z2_acc = 0.0
            n_tot = 0

            for layer in self.model.layers:
                m = layer.m
                v = torch.clamp(layer.v, min=1e-12)

                m_tmp = m.detach().clone().requires_grad_(True)
                v_tmp = v.detach().clone().requires_grad_(True)

                sigma2 = s2 + v_tmp
                lp = -0.5 * (m_tmp ** 2) / sigma2 - 0.5 * torch.log(2.0 * math.pi * sigma2)
                logZ_prior = lp.sum()

                for p in (m_tmp, v_tmp):
                    if p.grad is not None:
                        p.grad.zero_()
                logZ_prior.backward()

                gm = m_tmp.grad
                gv = v_tmp.grad

                m_new = m + v * gm
                v_new = v - (v * v) * (gm * gm - 2.0 * gv)
                v_new = torch.clamp(v_new, min=1e-12, max=1e3)

                with torch.no_grad():
                    layer.m.copy_(m_new.detach())
                    layer.v.copy_(v_new.detach())

                def sum_logZ_given_s2(s2_local: torch.Tensor) -> torch.Tensor:
                    sigma2_local = s2_local + v
                    lp_local = -0.5 * (m ** 2) / sigma2_local - 0.5 * torch.log(2.0 * math.pi * sigma2_local)
                    return lp_local.sum()

                Z_acc += sum_logZ_given_s2(s2).detach()
                Z1_acc += sum_logZ_given_s2(s2_1).detach()
                Z2_acc += sum_logZ_given_s2(s2_2).detach()
                n_tot += m.numel()

            denom = max(n_tot, 1)
            logZ_avg = Z_acc / denom
            logZ1_avg = Z1_acc / denom
            logZ2_avg = Z2_acc / denom

            alpha_l, beta_l = self._gamma_adf_update_from_Z(logZ_avg, logZ1_avg, logZ2_avg, alpha_l, beta_l)

        self.alpha_l = alpha_l.detach()
        self.beta_l = beta_l.detach()

    def _collect_dataset(self, loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        for batch_X, batch_y in loader:
            xs.append(batch_X)
            ys.append(batch_y)
        X_full = torch.cat(xs, dim=0).to(device=self.device, dtype=self.dtype)
        y_full = torch.cat(ys, dim=0).to(device=self.device, dtype=self.dtype)
        return X_full, y_full

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            num_epochs: int = 100,
            step_clip: Optional[float] = 2.0,
            prior_refresh: int = 1,
            **kwargs) -> Dict[str, List[float]]:

        history: Dict[str, List[float]] = {'train_rmse': [], 'train_nlpd': []}
        if val_loader is not None:
            history['val_rmse'] = []
            history['val_nlpd'] = []

        for epoch in range(num_epochs):
            X_full, y_full = self._collect_dataset(train_loader)
            order = torch.randperm(X_full.shape[0], device=self.device)
            for idx in order.tolist():
                x = X_full[idx]
                y = y_full[idx]
                self._single_datapoint_adf_step(x, y, step_clip)

            if prior_refresh > 0:
                self._prior_refresh_epoch(n_refresh=prior_refresh)

            train_rmse, train_nlpd = self._evaluate_loader(train_loader)
            history['train_rmse'].append(train_rmse)
            history['train_nlpd'].append(train_nlpd)

            if val_loader is not None:
                val_rmse, val_nlpd = self._evaluate_loader(val_loader)
                history['val_rmse'].append(val_rmse)
                history['val_nlpd'].append(val_nlpd)

        self.is_fitted = True
        return history

    def _evaluate_loader(self, loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        preds = []
        targets = []
        nlpd_sum = 0.0
        count = 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device=self.device, dtype=self.dtype)
                batch_y = batch_y.to(device=self.device, dtype=self.dtype).view(batch_X.shape[0], -1)
                mean, var = self._predictive_mean_var(batch_X)
                diff = batch_y - mean
                nlpd_batch = 0.5 * torch.log(2.0 * math.pi * var) + 0.5 * (diff * diff) / var
                nlpd_sum += nlpd_batch.sum().item()
                count += batch_y.numel()
                preds.append(mean.detach().cpu())
                targets.append(batch_y.detach().cpu())

        preds_all = torch.cat(preds, dim=0)
        targets_all = torch.cat(targets, dim=0)
        rmse = torch.sqrt(torch.mean((preds_all - targets_all) ** 2)).item()
        nlpd = nlpd_sum / max(count, 1)
        return rmse, nlpd

    def _predictive_mean_var(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mz, vz = self.model.forward_moments(X)
        alpha = torch.clamp(self.alpha_g, min=1.0 + 1e-6)
        noise_var = self.beta_g / (alpha - 1.0)
        var = torch.clamp(noise_var + vz, min=1e-12)
        return mz, var

    def predict(self, X: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        mean, var = self._predictive_mean_var(X.to(device=self.device, dtype=self.dtype))
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            (n_samples,) + mean.shape, device=self.device, dtype=self.dtype
        )
        return mean.detach(), samples.detach()

    def noise_variance(self) -> torch.Tensor:
        alpha = torch.clamp(self.alpha_g, min=1.0 + 1e-6)
        return self.beta_g / (alpha - 1.0)

    def sample_models(self, n_models: int = 10) -> List[nn.Module]:
        models = []
        for _ in range(n_models):
            model_copy = self._sample_single_model()
            models.append(model_copy)
        return models

    def _sample_single_model(self) -> nn.Module:
        layers: List[nn.Module] = []
        for li, layer in enumerate(self.model.layers):
            d = layer.in_features + 1
            scale = 1.0 / math.sqrt(d)
            weight_full = torch.normal(
                layer.m.detach(), torch.sqrt(torch.clamp(layer.v.detach(), min=1e-12))
            ) * scale
            weight = weight_full[:, :layer.in_features]
            bias = weight_full[:, layer.in_features]
            linear = nn.Linear(layer.in_features, layer.out_features)
            linear = linear.to(device=self.device, dtype=self.dtype)
            linear.weight.data = weight.to(device=self.device, dtype=self.dtype)
            linear.bias.data = bias.to(device=self.device, dtype=self.dtype)
            layers.append(linear)
            if li < len(self.model.layers) - 1:
                layers.append(nn.ReLU())
        model = nn.Sequential(*layers)
        model.eval()
        return model

    def _get_ensemble_state(self) -> Dict[str, Any]:
        return {
            'alpha_g': self.alpha_g,
            'beta_g': self.beta_g,
            'alpha_l': self.alpha_l,
            'beta_l': self.beta_l,
            'dtype': self.dtype,
        }

    def _set_ensemble_state(self, state: Dict[str, Any]):
        self.alpha_g = state.get('alpha_g', self.alpha_g).to(self.device)
        self.beta_g = state.get('beta_g', self.beta_g).to(self.device)
        self.alpha_l = state.get('alpha_l', self.alpha_l).to(self.device)
        self.beta_l = state.get('beta_l', self.beta_l).to(self.device)
