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
