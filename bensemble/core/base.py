import abc
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class BaseBayesianEnsemble(abc.ABC):
    """Базовый класс для всех методов байесовского ансамблирования"""
    '''The base class for all Bayesian ensembling methods'''

    def __init__(self, model: nn.Module, **kwargs):
        self.model = model
        self.is_fitted = False
        self.ensemble = []

    @abc.abstractmethod
    def fit(
        self,  
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """Обучение ансамбля"""
        """Ensemble training"""
        ...

    @abc.abstractmethod
    def predict(
        self, X: torch.Tensor, n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Предсказание с оценкой неопределенности"""
        """Prediction with uncertainty estimation"""
        ...

    @abc.abstractmethod
    def sample_models(
        self, n_models: int = 10
    ) -> List[
        nn.Module
    ]:  
        """Сэмплирование моделей из апостериорного распределения"""
        """Sampling models from a posteriori distribution"""
        ...

    def save(self, path: str):  
        """Сохранение обученного ансамбля"""
        """Saving a trained ensemble"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "ensemble_state": self._get_ensemble_state(),
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path: str):  
        """Загрузка обученного ансамбля"""
        """Loading a trained ensemble"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._set_ensemble_state(checkpoint["ensemble_state"])
        self.is_fitted = checkpoint["is_fitted"]

    @abc.abstractmethod
    def _get_ensemble_state(self) -> Dict[str, Any]:
        """Получение внутреннего состояния ансамбля"""
        """Getting the internal state of the ensemble"""
        ...

    @abc.abstractmethod
    def _set_ensemble_state(self, state: Dict[str, Any]):
        """Установка внутреннего состояния ансамбля"""
        """Setting the internal state of the ensemble"""
        ...
