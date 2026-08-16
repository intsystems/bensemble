from collections.abc import Callable

import torch
from torch import nn

from bensemble.core.member import ExplicitMembers, MemberAdapter, StochasticMembers
from bensemble.core.types import MemberPredictions, PosteriorSource, Predictions


class Ensemble(nn.Module):
    """Ensemble of model predictors combined via an aggregation function."""

    def __init__(
        self,
        members: MemberAdapter,
        combiner: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """Initializes the Ensemble container.

        Args:
            members: MemberAdapter instance wrapping ensemble members.
            combiner: Aggregation function taking (M, batch, *) and returning
                combined predictions (batch, *). Defaults to averaging across members.
        """
        super().__init__()
        self.members = members
        self.combiner = combiner or (lambda preds: preds.mean(dim=0))

    def predict_members(self, x: torch.Tensor) -> MemberPredictions:
        """Computes predictions for each individual member.

        Args:
            x: Input tensor of shape (batch_size, *).

        Returns:
            MemberPredictions: Per-member predictions of shape (num_members, batch_size, *).
        """
        return self.members.predict_all(x)

    def forward(self, x: torch.Tensor) -> Predictions:
        """Computes combined ensemble prediction.

        Args:
            x: Input tensor of shape (batch_size, *).

        Returns:
            Predictions: Aggregated output prediction tensor.
        """
        return self.combiner(self.predict_members(x))

    @property
    def num_members(self) -> int:
        """int: Number of ensemble members."""
        return self.members.size

    @property
    def member_modules(self) -> list[nn.Module]:
        """list[nn.Module]: Underlying member modules."""
        return self.members.modules

    @classmethod
    def from_models(cls, models: list[nn.Module]) -> "Ensemble":
        """Creates an explicit ensemble from a list of independent models.

        Args:
            models: List of trained PyTorch modules.

        Returns:
            Ensemble: Configured Ensemble instance.
        """
        return cls(members=ExplicitMembers(models))

    @classmethod
    def from_stochastic(
        cls, model: nn.Module, num_samples: int = 30, mode: str = "auto"
    ) -> "Ensemble":
        """Creates an implicit ensemble from a single stochastic model.

        Args:
            model: Stochastic neural network module.
            num_samples: Number of stochastic forward samples. Defaults to 30.
            mode: Stochastic mode ("dropout", "bayesian", or "auto"). Defaults to "auto".

        Returns:
            Ensemble: Configured Ensemble instance.
        """
        return cls(members=StochasticMembers(model, num_samples, mode))

    @classmethod
    def from_posterior(
        cls, source: PosteriorSource, n_members: int = 10, **kwargs
    ) -> "Ensemble":
        """Creates an explicit ensemble by sampling from an approximated posterior.

        Args:
            source: Posterior approximation source implementing `sample_models`.
            n_members: Number of members to sample. Defaults to 10.
            **kwargs: Additional keyword arguments passed to `sample_models`.

        Returns:
            Ensemble: Configured Ensemble instance.
        """
        models = source.sample_models(n_members, **kwargs)
        return cls(members=ExplicitMembers(models))
