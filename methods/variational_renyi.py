import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from ..core.base import BaseBayesianEnsemble

class VariationalRenyi(BaseBayesianEnsemble):
    """
    Algorithm 3: Variational Renyi Bound (VR)
    """
    
    def __init__(self, 
                 model: nn.Module,
                 alpha: float = 1.0,  # α → 1 дает стандартный VI
                 prior_sigma: float = 1.0,
                 initial_rho: float = -3.0,
                 **kwargs):
        super().__init__(model)
        self.alpha = alpha
        self.prior_sigma = prior_sigma
        self.initial_rho = initial_rho
        
        # Преобразование модели аналогично VariationalInference
        self.bayesian_model = self._make_bayesian(model)
        self.optimizer = None
    
    def _make_bayesian(self, model: nn.Module) -> nn.Module:
        """Аналогично VariationalInference"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight_mu = nn.Parameter(module.weight.data.clone())
                module.weight_rho = nn.Parameter(torch.full_like(module.weight, self.initial_rho))
                module.bias_mu = nn.Parameter(module.bias.data.clone()) if module.bias is not None else None
                module.bias_rho = nn.Parameter(torch.full_like(module.bias, self.initial_rho)) if module.bias is not None else None
                
                del module.weight, module.bias
        return model
    
    def _reparameterize(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = torch.log1p(torch.exp(rho))
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    
    def _compute_log_weights(self, batch_X: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        """Вычисление логарифма весов для VR bound"""
        # Лог правдоподобия
        output = self.forward(batch_X)
        if output.dim() > 1:  # Классификация
            log_likelihood = -F.cross_entropy(output, batch_y, reduction='none')
        else:  # Регрессия
            log_likelihood = -F.mse_loss(output, batch_y, reduction='none')
        
        # Лог априорного распределения
        log_prior = 0.0
        for module in self.bayesian_model.modules():
            if hasattr(module, 'weight_mu'):
                weight = getattr(module, 'weight', None)
                if weight is not None:
                    log_prior += torch.distributions.Normal(0, self.prior_sigma).log_prob(weight).sum()
                
                bias = getattr(module, 'bias', None)
                if bias is not None:
                    log_prior += torch.distributions.Normal(0, self.prior_sigma).log_prob(bias).sum()
        
        # Лог вариационного распределения
        log_variational = 0.0
        for module in self.bayesian_model.modules():
            if hasattr(module, 'weight_mu'):
                weight_sigma = torch.log1p(torch.exp(module.weight_rho))
                log_variational += torch.distributions.Normal(module.weight_mu, weight_sigma).log_prob(module.weight).sum()
                
                if hasattr(module, 'bias_mu') and module.bias_mu is not None:
                    bias_sigma = torch.log1p(torch.exp(module.bias_rho))
                    log_variational += torch.distributions.Normal(module.bias_mu, bias_sigma).log_prob(module.bias).sum()
        
        return log_likelihood + log_prior - log_variational
    
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
            n_samples: int = 5,
            **kwargs) -> Dict[str, List[float]]:
        
        self.optimizer = torch.optim.Adam(self.bayesian_model.parameters(), lr=lr)
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            self.bayesian_model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Многократное сэмплирование для оценки VR bound
                log_weights = []
                for _ in range(n_samples):
                    log_w = self._compute_log_weights(batch_X, batch_y)
                    log_weights.append(log_w.unsqueeze(-1))
                
                log_weights = torch.cat(log_weights, dim=-1)
                
                # VR bound
                if self.alpha == 1:
                    # Стандартный ELBO
                    loss = -log_weights.mean()
                else:
                    # Renyi bound
                    loss = - (1/(1-self.alpha)) * torch.logsumexp((1-self.alpha) * log_weights, dim=-1) + torch.log(torch.tensor(n_samples, dtype=torch.float))
                    loss = loss.mean()
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            history['train_loss'].append(train_loss / len(train_loader))
            
            if val_loader is not None:
                val_loss = self._validate(val_loader, n_samples)
                history['val_loss'].append(val_loss)
        
        self.is_fitted = True
        return history
    
    def _validate(self, val_loader: torch.utils.data.DataLoader, n_samples: int) -> float:
        self.bayesian_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                log_weights = []
                for _ in range(n_samples):
                    log_w = self._compute_log_weights(batch_X, batch_y)
                    log_weights.append(log_w.unsqueeze(-1))
                
                log_weights = torch.cat(log_weights, dim=-1)
                
                if self.alpha == 1:
                    loss = -log_weights.mean()
                else:
                    loss = - (1/(1-self.alpha)) * torch.logsumexp((1-self.alpha) * log_weights, dim=-1) + torch.log(torch.tensor(n_samples, dtype=torch.float))
                    loss = loss.mean()
                
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
            model_copy = self._sample_single_model()
            models.append(model_copy)
        return models
    
    def _sample_single_model(self) -> nn.Module:
        model_copy = self._make_bayesian(self.model.__class__())
        
        with torch.no_grad():
            for module_orig, module_copy in zip(self.bayesian_model.modules(), model_copy.modules()):
                if hasattr(module_orig, 'weight_mu'):
                    weight = self._reparameterize(module_orig.weight_mu, module_orig.weight_rho)
                    module_copy.weight.data = weight.data
                    
                    if hasattr(module_orig, 'bias_mu') and module_orig.bias_mu is not None:
                        bias = self._reparameterize(module_orig.bias_mu, module_orig.bias_rho)
                        module_copy.bias.data = bias.data
        
        return model_copy
    
    def _get_ensemble_state(self) -> Dict[str, Any]:
        return {
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'alpha': self.alpha
        }
    
    def _set_ensemble_state(self, state: Dict[str, Any]):
        if self.optimizer and state['optimizer_state']:
            self.optimizer.load_state_dict(state['optimizer_state'])
        self.alpha = state.get('alpha', 1.0)