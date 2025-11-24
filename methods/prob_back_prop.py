import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..core.base import BaseBayesianEnsemble

class PBPEnsemble(BaseBayesianEnsemble):
    """Bayesian ensemble using Probabilistic Backpropagation (PBP)"""
    
    class ProbLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            d = self.in_features + 1  # +1 bias
            h = self.out_features
            scale = 1.0 / math.sqrt(d)
            self.m = nn.Parameter(scale * torch.randn(h, d, dtype=torch.float64))
            self.v = nn.Parameter(0.5 * torch.ones(h, d, dtype=torch.float64))

    class PBPNet(nn.Module):
        def __init__(self, layer_sizes: List[int]):
            super().__init__()
            self.layers: List[PBPEnsemble.ProbLinear] = nn.ModuleList()
            for i in range(len(layer_sizes) - 1):
                self.layers.append(PBPEnsemble.ProbLinear(layer_sizes[i], layer_sizes[i+1]))

        def forward_moments(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            batch = x.shape[0]
            mz = torch.cat([x, torch.ones(batch, 1, dtype=torch.float64)], dim=1)
            vz = torch.zeros_like(mz)
            
            for li, layer in enumerate(self.layers):
                d = layer.in_features + 1
                scale = 1.0 / math.sqrt(d)
                
                ma = (mz @ layer.m.t()) * scale
                term1 = (vz @ (layer.m ** 2).t())
                term2 = ((mz ** 2) @ layer.v.t())
                term3 = (vz @ layer.v.t())
                va = (term1 + term2 + term3) * (scale ** 2)
                
                is_last = (li == len(self.layers) - 1)
                if not is_last:
                    mb, vb = PBPEnsemble.relu_moments(ma, va)
                    mz = torch.cat([mb, torch.ones(batch, 1, dtype=torch.float64)], dim=1)
                    vz = torch.cat([vb, torch.zeros(batch, 1, dtype=torch.float64)], dim=1)
                else:
                    mz = ma
                    vz = va
            return mz, vz

    def __init__(self, layer_sizes: List[int], **kwargs):
        model = self.PBPNet(layer_sizes)
        super().__init__(model, **kwargs)
        self.alpha_g = torch.tensor(6.0, dtype=torch.float64)  # noise precision
        self.beta_g  = torch.tensor(6.0, dtype=torch.float64)
        self.alpha_l = torch.tensor(6.0, dtype=torch.float64)  # weight precision
        self.beta_l  = torch.tensor(6.0, dtype=torch.float64)

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def Phi(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    @staticmethod
    def relu_moments(m: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
        v = torch.clamp(v, min=eps)
        sigma = torch.sqrt(v)
        alpha = m / sigma
        alpha_eval = torch.clamp(alpha, min=-10.0, max=10.0)
        pdf = PBPEnsemble.phi(alpha_eval)
        cdf = PBPEnsemble.Phi(alpha_eval)
        mean = sigma * pdf + m * cdf
        second_moment = (v + m * m) * cdf + m * sigma * pdf
        var = torch.clamp(second_moment - mean * mean, min=eps)
        return mean, var

    @staticmethod
    def logZ_gaussian_likelihood(y, mz, vz, alpha_g, beta_g, eps=1e-12):
        alpha_g = torch.clamp(alpha_g, min=1.0 + 1e-6)
        sigma2_eff = beta_g / (alpha_g - 1.0) + vz
        sigma2_eff = torch.clamp(sigma2_eff, min=eps)
        return (-0.5 * ((y - mz) ** 2) / sigma2_eff - 0.5 * torch.log(2.0 * math.pi * sigma2_eff)).squeeze(-1)

    @staticmethod
    def gamma_adf_update_from_Z(logZ, logZ1, logZ2, alpha_old, beta_old, clamp_eps=1e-9):
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

    def single_datapoint_adf_step(self, x: torch.Tensor, y: torch.Tensor, step_clip: float = 1.0):
        net = self.model
        for layer in net.layers:
            layer.m.requires_grad_(True)
            layer.v.requires_grad_(True)
        mz, vz = net.forward_moments(x[None, :])
        logZ = self.logZ_gaussian_likelihood(y[None, None], mz, vz, self.alpha_g, self.beta_g).mean()
        logZ1 = self.logZ_gaussian_likelihood(y[None, None], mz, vz, self.alpha_g + 1.0, self.beta_g).mean()
        logZ2 = self.logZ_gaussian_likelihood(y[None, None], mz, vz, self.alpha_g + 2.0, self.beta_g).mean()
        for p in net.parameters():
            if p.grad is not None:
                p.grad.zero_()
        logZ.backward()
        for layer in net.layers:
            gm = layer.m.grad
            gv = layer.v.grad
            if step_clip is not None:
                gm = torch.clamp(gm, min=-step_clip, max=step_clip)
                gv = torch.clamp(gv, min=-step_clip, max=step_clip)
            m = layer.m
            v = torch.clamp(layer.v, min=1e-10, max=1e3)
            m_new = m + v * gm
            v_new = v - (v * v) * (gm * gm - 2.0 * gv)
            m_new = m_new.detach()
            v_new = v_new.detach()
            with torch.no_grad():
                layer.m.copy_(m_new)
                layer.v.copy_(v_new)
        self.alpha_g, self.beta_g = self.gamma_adf_update_from_Z(logZ.detach(), logZ1.detach(), logZ2.detach(), self.alpha_g, self.beta_g)
        for layer in net.layers:
            layer.m.grad = None
            layer.v.grad = None

    def prior_refresh_epoch(self, n_refresh: int = 1):
        net = self.model
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
            for layer in net.layers:
                m = layer.m
                v = torch.clamp(layer.v, min=1e-12)
                m_tmp = m.detach().clone().requires_grad_(True)
                v_tmp = v.detach().clone().requires_grad_(True)
                sigma2 = s2 + v_tmp
                lp = -0.5 * (m_tmp ** 2) / sigma2 - 0.5 * torch.log(2.0 * math.pi * sigma2)
                logZ_prior = lp.sum()
                for p in [m_tmp, v_tmp]:
                    if p.grad is not None:
                        p.grad.zero_()
                logZ_prior.backward()
                gm = m_tmp.grad
                gv = v_tmp.grad
                m_new = m + v * gm
                v_new = v - (v * v) * (gm * gm - 2.0 * gv)
                v_new = torch.clamp(v_new, min=1e-12, max=1e3)
                m_new = m_new.detach()
                v_new = v_new.detach()
                with torch.no_grad():
                    layer.m.copy_(m_new)
                    layer.v.copy_(v_new)
                def sum_logZ_given_s2(s2_local: torch.Tensor) -> torch.Tensor:
                    sigma2_local = s2_local + v
                    lp_local = -0.5 * (m ** 2) / sigma2_local - 0.5 * torch.log(2.0 * math.pi * sigma2_local)
                    return lp_local.sum()
                Z_acc += sum_logZ_given_s2(s2).detach()
                Z1_acc += sum_logZ_given_s2(s2_1).detach()
                Z2_acc += sum_logZ_given_s2(s2_2).detach()
                n_tot += m.numel()
            logZ_avg = Z_acc / max(n_tot, 1)
            logZ1_avg = Z1_acc / max(n_tot, 1)
            logZ2_avg = Z2_acc / max(n_tot, 1)
            alpha_l, beta_l = self.gamma_adf_update_from_Z(logZ_avg, logZ1_avg, logZ2_avg, alpha_l, beta_l)
        self.alpha_l = alpha_l.detach()
        self.beta_l = beta_l.detach()

    def fit(self, 
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            epochs: int = 1,
            step_clip: float = 2.0,
            refresh_prior: bool = True,
            **kwargs) -> Dict[str, List[float]]:
        """Пример реализации fit для PBP"""
        self.model.train()
        history = {'rmse': [], 'nlpd': []}
        for epoch in range(epochs):
            for x, y in train_loader:
                self.single_datapoint_adf_step(x, y, step_clip=step_clip)
            if refresh_prior:
                self.prior_refresh_epoch(n_refresh=1)
        self.is_fitted = True
        return history

    def predict(self, X: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Предсказание среднего и дисперсии"""
        self.model.eval()
        with torch.no_grad():
            mz, vz = self.model.forward_moments(X)
            sigma2_eff = self.beta_g / (self.alpha_g - 1.0) + vz
        return mz, sigma2_eff

    def sample_models(self, n_models: int = 10) -> List[nn.Module]:
        """Простейший метод сэмплирования моделей — копии текущей модели"""
        return [self.model for _ in range(n_models)]

    def _get_ensemble_state(self) -> Dict[str, Any]:
        return {
            'alpha_g': self.alpha_g,
            'beta_g': self.beta_g,
            'alpha_l': self.alpha_l,
            'beta_l': self.beta_l,
            'model_state': self.model.state_dict()
        }

    def _set_ensemble_state(self, state: Dict[str, Any]):
        self.alpha_g = state['alpha_g']
        self.beta_g = state['beta_g']
        self.alpha_l = state['alpha_l']
        self.beta_l = state['beta_l']
        self.model.load_state_dict(state['model_state'])
