import torch.nn as nn
import torch
from bensemble.core.member import MemberAdapter, ExplicitMembers, StochasticMembers
from bensemble.core.types import Predictions, MemberPredictions, PosteriorSource


class Ensemble(nn.Module):
    """
    A collection of predictors whose outputs are combined.
    Ensemble doesn't care where its members came from.
    """

    def __init__(self, members: MemberAdapter, combiner=None):
        super().__init__()
        self.members = members
        self.combiner = combiner or (lambda preds: preds.mean(dim=0))

    def predict_members(self, x: torch.Tensor) -> MemberPredictions:
        """(M, batch, *) — raw per-member outputs."""
        return self.members.predict_all(x)

    def forward(self, x: torch.Tensor) -> Predictions:
        """Combined prediction."""
        return self.combiner(self.predict_members(x))

    @property
    def num_members(self) -> int:
        return self.members.size

    @property
    def member_modules(self) -> list[nn.Module]:
        """Access underlying nn.Modules (for KL, parameters, etc.)."""
        return self.members.modules

    @classmethod
    def from_models(cls, models: list[nn.Module]) -> "Ensemble":
        """Explicit ensemble from a list of independent models."""
        return cls(members=ExplicitMembers(models))

    @classmethod
    def from_stochastic(
        cls, model: nn.Module, num_samples: int = 30, mode: str = "auto"
    ) -> "Ensemble":
        """
        Implicit ensemble from a model with stochastic forward passes.
        """
        return cls(members=StochasticMembers(model, num_samples, mode))

    @classmethod
    def from_posterior(
        cls, source: PosteriorSource, n_members: int = 10, **kwargs
    ) -> "Ensemble":
        """
        Sample an explicit ensemble from a fitted posterior approximation.
        """
        models = source.sample_models(n_members, **kwargs)
        return cls(members=ExplicitMembers(models))
