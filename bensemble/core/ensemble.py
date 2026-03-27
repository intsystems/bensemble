import torch.nn as nn
import torch
from member import MemberAdapter
from types import Predictions, MemberPredictions


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
