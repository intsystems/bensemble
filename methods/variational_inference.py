import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from ..core.base import BaseBayesianEnsemble

class VariationalInference(BaseBayesianEnsemble):
    """
    Algorithm 1: Practical Variational Inference (Graves)
    """
    
    def __init__(self, 
                 model: nn.Module,
                 prior_sigma: float = 1.0,
                 initial_rho: float = -3.0,
                 **kwargs):
        super().__init__(model)
        self.prior_sigma = prior_sigma
        self.initial_rho = initial_rho
        
        # Преобразование модели в байесовскую
        self.bayesian_model = self._make_bayesian(model)
        self.optimizer = None
        
    def _make_bayesian(self, model: nn.Module) -> nn.Module:
        """Преобразование обычной модели в байесовскую с вариационными параметрами"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight_mu = nn.Parameter(module.weight.data.clone())
                module.weight_rho = nn.Parameter(torch.full_like(module.weight, self.initial_rho))
                module.bias_mu = nn.Parameter(module.bias.data.clone()) if module.bias is not None else None
                module.bias_rho = nn.Parameter(torch.full_like(module.bias, self.initial_rho)) if module.bias is not None else None
                
                # Удаляем оригинальные параметры
                del module.weight, module.bias
                
        return model
    
    def _reparameterize(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        sigma = torch.log1p(torch.exp(rho))
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    
    def _kl_divergence(self) -> torch.Tensor:
        """Вычисление KL дивергенции"""
        kl = 0.0
        for module in self.bayesian_model.modules():
            if hasattr(module, 'weight_mu'):
                # KL для весов
                w_sigma = torch.log1p(torch.exp(module.weight_rho))
                kl += torch.sum(torch.log(self.prior_sigma / w_sigma) + 
                               (w_sigma**2 + module.weight_mu**2) / (2 * self.prior_sigma**2) - 0.5)
                
                # KL для bias
                if hasattr(module, 'bias_mu') and module.bias_mu is not None:
                    b_sigma = torch.log1p(torch.exp(module.bias_rho))
                    kl += torch.sum(torch.log(self.prior_sigma / b_sigma) + 
                                   (b_sigma**2 + module.bias_mu**2) / (2 * self.prior_sigma**2) - 0.5)
        return kl
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход с репараметризацией"""
        for module in self.bayesian_model.modules():
            if hasattr(module, 'weight_mu'):
                module.weight = self._reparameterize(module.weight_mu, module.weight_rho)
                if hasattr(module, 'bias_mu') and module.bias_mu is not None:
                    module.bias = self._reparameterize(module.bias_mu, module.bias_rho)
        return self.bayesian_model(x)
    
    def fit(self, 
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            num_epochs: int = 100,
            lr: float = 1e-3,
            kl_weight: float = 1.0,
            **kwargs) -> Dict[str, List[float]]:
        
        self.optimizer = torch.optim.Adam(self.bayesian_model.parameters(), lr=lr)
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            # Обучение
            self.bayesian_model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Прямой проход
                output = self.forward(batch_X)
                
                # Вычисление лосса (negative ELBO)
                likelihood = F.cross_entropy(output, batch_y) if output.dim() > 1 else F.mse_loss(output, batch_y)
                kl = self._kl_divergence()
                loss = likelihood + kl_weight * kl / len(train_loader.dataset)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            history['train_loss'].append(train_loss / len(train_loader))
            
            # Валидация
            if val_loader is not None:
                val_loss = self._validate(val_loader, kl_weight)
                history['val_loss'].append(val_loss)
            
        self.is_fitted = True
        return history
    
    def _validate(self, val_loader: torch.utils.data.DataLoader, kl_weight: float) -> float:
        self.bayesian_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                output = self.forward(batch_X)
                likelihood = F.cross_entropy(output, batch_y) if output.dim() > 1 else F.mse_loss(output, batch_y)
                kl = self._kl_divergence()
                loss = likelihood + kl_weight * kl / len(val_loader.dataset)
                val_loss += loss.item()
        return val_loss / len(val_loader)
    
    def predict(self, X: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        self.bayesian_model.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(X)
                predictions.append(output.unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0)
        mean_prediction = predictions.mean(dim=0)
        
        return mean_prediction, predictions
    
    def sample_models(self, n_models: int = 10) -> List[nn.Module]:
        models = []
        for i in range(n_models):
            # Создаем копию модели с зафиксированными сэмплированными весами
            model_copy = self._sample_single_model()
            models.append(model_copy)
        return models
    
    def _sample_single_model(self) -> nn.Module:
        """Сэмплирование одной модели с фиксированными весами"""
        model_copy = self._make_bayesian(self.model.__class__())
        
        with torch.no_grad():
            for module_orig, module_copy in zip(self.bayesian_model.modules(), model_copy.modules()):
                if hasattr(module_orig, 'weight_mu'):
                    # Сэмплируем веса
                    weight = self._reparameterize(module_orig.weight_mu, module_orig.weight_rho)
                    module_copy.weight.data = weight.data
                    
                    if hasattr(module_orig, 'bias_mu') and module_orig.bias_mu is not None:
                        bias = self._reparameterize(module_orig.bias_mu, module_orig.bias_rho)
                        module_copy.bias.data = bias.data
        
        return model_copy
    
    def pruning_heuristic(self, threshold: float = 0.83) -> nn.Module:
        """Эвристика прунинга на основе отношения |mu|/sigma"""
        pruned_model = self.model.__class__()
        
        with torch.no_grad():
            for module_orig, module_pruned in zip(self.bayesian_model.modules(), pruned_model.modules()):
                if hasattr(module_orig, 'weight_mu'):
                    # Вычисляем отношение |mu|/sigma
                    weight_sigma = torch.log1p(torch.exp(module_orig.weight_rho))
                    weight_ratio = torch.abs(module_orig.weight_mu) / weight_sigma
                    
                    # Маска для прунинга
                    mask = weight_ratio > threshold
                    module_pruned.weight.data = module_orig.weight_mu.data * mask.float()
                    
                    if hasattr(module_orig, 'bias_mu') and module_orig.bias_mu is not None:
                        bias_sigma = torch.log1p(torch.exp(module_orig.bias_rho))
                        bias_ratio = torch.abs(module_orig.bias_mu) / bias_sigma
                        bias_mask = bias_ratio > threshold
                        module_pruned.bias.data = module_orig.bias_mu.data * bias_mask.float()
        
        return pruned_model
    
    def _get_ensemble_state(self) -> Dict[str, Any]:
        return {
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None
        }
    
    def _set_ensemble_state(self, state: Dict[str, Any]):
        if self.optimizer and state['optimizer_state']:
            self.optimizer.load_state_dict(state['optimizer_state'])