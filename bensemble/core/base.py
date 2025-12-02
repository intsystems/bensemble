import abc
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class BaseBayesianEnsemble(abc.ABC):
    """Базовый класс для всех методов байесовского ансамблирования"""

    def __init__(self, model: nn.Module, **kwargs):
        self.model = model
        self.is_fitted = False
        self.ensemble = []

    @abc.abstractmethod
    def fit(
        self,  #
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """Обучение ансамбля"""
        # Надо обсудить параметры и реализацию
        ...

    @abc.abstractmethod
    def predict(
        self, X: torch.Tensor, n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Предсказание с оценкой неопределенности"""
        # Надо обсудить параметры и реализацию
        ...

    @abc.abstractmethod
    def sample_models(
        self, n_models: int = 10
    ) -> List[
        nn.Module
    ]:  # Добавить онлайн-поддерживание сгенереированных моделей -> Поменять предикт
        """Сэмплирование моделей из апостериорного распределения"""
        # Надо обсудить параметры
        ...

    def save(self, path: str):  # Сделать загрузку и выгрузку моделей через стэйт диктов
        """Сохранение обученного ансамбля"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "ensemble_state": self._get_ensemble_state(),
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path: str):  # Сделать загрузку и выгрузку моделей через стэйт диктов
        """Загрузка обученного ансамбля"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._set_ensemble_state(checkpoint["ensemble_state"])
        self.is_fitted = checkpoint["is_fitted"]

    @abc.abstractmethod
    def _get_ensemble_state(self) -> Dict[str, Any]:
        """Получение внутреннего состояния ансамбля"""
        # Надо обсудить параметры
        ...

    @abc.abstractmethod
    def _set_ensemble_state(self, state: Dict[str, Any]):
        """Установка внутреннего состояния ансамбля"""
        # Надо обсудить параметры
        ...
