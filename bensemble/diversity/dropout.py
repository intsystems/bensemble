from torch import nn

from ..core.ensemble import Ensemble


class MCDropoutEnsembler:
    """Wrapper for building Monte Carlo Dropout ensembles from trained models."""

    def __init__(self, model: nn.Module):
        """Initializes the MCDropoutEnsembler.

        Args:
            model: Neural network model containing dropout layers.
        """
        self.model = model

    def build_ensemble(self, num_samples: int = 30) -> Ensemble:
        """Builds an Ensemble module utilizing MC Dropout forward passes.

        Args:
            num_samples: Number of stochastic forward passes per prediction. Defaults to 30.

        Returns:
            Ensemble: Ensemble instance wrapping the stochastic model.
        """
        return Ensemble.from_stochastic(
            self.model, num_samples=num_samples, mode="dropout"
        )
