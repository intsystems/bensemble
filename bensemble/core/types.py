from typing import Protocol, runtime_checkable
import torch
import torch.nn as nn


@runtime_checkable
class Predictor(Protocol):
    """Anything that maps input -> output."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class KLProvider(Protocol):
    """Any module that can report its KL divergence."""

    def kl_divergence(self) -> torch.Tensor: ...


@runtime_checkable
class PosteriorSource(Protocol):
    """Any method that can sample models from an approximate posterior."""

    def sample_models(self, n_models: int) -> list[nn.Module]: ...


# Type aliases
Predictions = torch.Tensor  # (batch, *output_shape)
MemberPredictions = torch.Tensor  # (M, batch, *output_shape)
