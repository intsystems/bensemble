from abc import abstractmethod
import torch.nn as nn
import torch
from types import MemberPredictions
from .layers.base import BaseBayesianLayer


class MemberAdapter(nn.Module):
    """Adapts different prediction sources into a uniform (M, batch, *) interface."""

    @abstractmethod
    def predict_all(self, x: torch.Tensor) -> MemberPredictions: ...

    @property
    @abstractmethod
    def size(self) -> int: ...

    @property
    @abstractmethod
    def modules(self) -> list[nn.Module]: ...


class ExplicitMembers(MemberAdapter):
    """Wraps a list of independent nn.Module instances."""

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def predict_all(self, x):
        return torch.stack([m(x) for m in self.models])

    @property
    def size(self):
        return len(self.models)

    @property
    def modules(self):
        return list(self.models)


class StochasticMembers(MemberAdapter):
    """
    Wraps a single model whose forward pass is stochastic.
    """

    ACTIVATORS = {
        "bayesian": "_activate_bayesian",
        "dropout": "_activate_dropout",
        "both": "_activate_both",
    }

    def __init__(self, model: nn.Module, num_samples: int = 30, mode: str = "auto"):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.mode = mode if mode != "auto" else self._detect_mode()

    def predict_all(self, x):
        was_training = self.model.training
        self._activate()
        with torch.no_grad():
            preds = torch.stack([self.model(x) for _ in range(self.num_samples)])
        self.model.train(was_training)
        return preds

    @property
    def size(self):
        return self.num_samples

    @property
    def modules(self):
        return [self.model]

    def _activate(self):
        activator = self.ACTIVATORS.get(self.mode)
        if activator:
            getattr(self, activator)()

    def _activate_bayesian(self):
        """Matches existing predict_with_uncertainty() pattern."""
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, BaseBayesianLayer):
                module.train()

    def _activate_dropout(self):
        """Matches existing enable_dropout() pattern."""
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _activate_both(self):
        self._activate_bayesian()
        self._activate_dropout()

    def _detect_mode(self):
        has_bayesian = any(
            isinstance(m, BaseBayesianLayer) for m in self.model.modules()
        )
        has_dropout = any(isinstance(m, nn.Dropout) for m in self.model.modules())
        if has_bayesian and has_dropout:
            return "both"
        elif has_bayesian:
            return "bayesian"
        elif has_dropout:
            return "dropout"
        return "dropout"
