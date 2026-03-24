from abc import abstractmethod
import torch.nn as nn
import torch
from types import MemberPredictions


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
