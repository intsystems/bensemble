import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from ..core.base import BaseBayesianEnsemble

class LaplaceApproximation(BaseBayesianEnsemble):
    """
    Algorithm 2: Scalable Laplace Approximation
    """
    
    def __init__(self, 
                 model: nn.Module,
                 prior_precision: float = 1.0,
                 **kwargs):
        super().__init__(model)
        self.prior_precision = prior_precision
        self.hessian_blocks = {}
        self.map_parameters = {}
        
    def fit(self, 
            train_loader: torch.utils.data.DataLoader,
            val_loader: None = None,
            num_epochs: int = 100,
            lr: float = 1e-3,
            **kwargs) -> Dict[str, List[float]]:
        
        # Сначала обучаем модель MAP оценке
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        history = {'train_loss': []}
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = F.cross_entropy(output, batch_y) if output.dim() > 1 else F.mse_loss(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            history['train_loss'].append(train_loss / len(train_loader))
        
        # Сохраняем MAP параметры
        self._save_map_parameters()
        
        # Вычисляем аппроксимацию Гессиана
        self._compute_hessian_approximation(train_loader)
        
        self.is_fitted = True
        return history
    
    def _save_map_parameters(self):
        """Сохранение MAP параметров"""
        for name, param in self.model.named_parameters():
            self.map_parameters[name] = param.data.clone()
    
    def _compute_hessian_approximation(self, train_loader: torch.utils.data.DataLoader):
        """Аппроксимация Гессиана методом Кронекера"""
        self.model.eval()
        
        # Для простоты используем диагональную аппроксимацию
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Диагональная аппроксимация Fisher information matrix
                fisher_diag = torch.zeros_like(param)
                
                for batch_X, batch_y in train_loader:
                    self.model.zero_grad()
                    output = self.model(batch_X)
                    
                    if output.dim() > 1:  # Классификация
                        prob = F.softmax(output, dim=1)
                        target = torch.multinomial(prob, 1).squeeze()
                        loss = F.cross_entropy(output, target)
                    else:  # Регрессия
                        loss = F.mse_loss(output, batch_y)
                    
                    loss.backward()
                    fisher_diag += param.grad.pow(2) if param.grad is not None else torch.zeros_like(param)
                
                fisher_diag /= len(train_loader)
                # Апостериорная точность = prior_precision + Fisher
                posterior_precision = self.prior_precision + fisher_diag
                self.hessian_blocks[name] = posterior_precision
    
    def predict(self, X: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        predictions = []
        
        for _ in range(n_samples):
            # Сэмплируем веса из лапласовского приближения
            self._sample_parameters()
            
            with torch.no_grad():
                output = self.model(X)
                predictions.append(output.unsqueeze(0))
            
            # Восстанавливаем MAP параметры
            self._restore_map_parameters()
        
        predictions = torch.cat(predictions, dim=0)
        mean_prediction = predictions.mean(dim=0)
        
        return mean_prediction, predictions
    
    def _sample_parameters(self):
        """Сэмплирование параметров из лапласовского приближения"""
        for name, param in self.model.named_parameters():
            if name in self.hessian_blocks:
                precision = self.hessian_blocks[name]
                covariance = 1.0 / (precision + 1e-8)  # Добавляем для численной стабильности
                sample = torch.randn_like(param) * torch.sqrt(covariance)
                param.data = self.map_parameters[name] + sample
    
    def _restore_map_parameters(self):
        """Восстановление MAP параметров"""
        for name, param in self.model.named_parameters():
            if name in self.map_parameters:
                param.data = self.map_parameters[name]
    
    def sample_models(self, n_models: int = 10) -> List[nn.Module]:
        models = []
        for i in range(n_models):
            model_copy = self._sample_single_model()
            models.append(model_copy)
        return models
    
    def _sample_single_model(self) -> nn.Module:
        model_copy = self.model.__class__()
        model_copy.load_state_dict(self.model.state_dict())
        
        with torch.no_grad():
            for name, param in model_copy.named_parameters():
                if name in self.hessian_blocks:
                    precision = self.hessian_blocks[name]
                    covariance = 1.0 / (precision + 1e-8)
                    sample = torch.randn_like(param) * torch.sqrt(covariance)
                    param.data = self.map_parameters[name] + sample
        
        return model_copy
    
    def _get_ensemble_state(self) -> Dict[str, Any]:
        return {
            'map_parameters': self.map_parameters,
            'hessian_blocks': self.hessian_blocks
        }
    
    def _set_ensemble_state(self, state: Dict[str, Any]):
        self.map_parameters = state['map_parameters']
        self.hessian_blocks = state['hessian_blocks']