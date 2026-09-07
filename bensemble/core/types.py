from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Predictor(Protocol):
    """Anything that maps input -> output."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Maps a batch of inputs to outputs."""


@runtime_checkable
class KLProvider(Protocol):
    """Any module that can report its KL divergence."""

    def kl_divergence(self) -> torch.Tensor:
        """Returns the KL divergence between the module's posterior and prior."""


@runtime_checkable
class PosteriorSource(Protocol):
    """Any method that can sample models from an approximate posterior."""

    def sample_models(self, n_models: int) -> list[nn.Module]:
        """Draws `n_models` networks from the approximate posterior."""


# Type aliases
Predictions = torch.Tensor  # (batch, *output_shape)
MemberPredictions = torch.Tensor  # (M, batch, *output_shape)
