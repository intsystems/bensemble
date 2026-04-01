from ..core.ensemble import Ensemble


class MCDropoutEnsembler:
    def __init__(self, model):
        self.model = model

    def build_ensemble(self, num_samples=30) -> Ensemble:
        """
        Returns:
            Ensemble: The final ensemble wrapped in bensemble's core abstraction.
        """
        return Ensemble.from_stochastic(self.model, num_samples=num_samples, mode="dropout")