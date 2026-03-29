from bensemble.core.ensemble import Ensemble


class MCDropoutEnsembler:
    def __init__(self):
        pass

    def build_ensemble(self) -> Ensemble:
        """
        Returns:
            Ensemble: The final ensemble wrapped in bensemble's core abstraction.
        """
        pass
        # return Ensemble.from_stochastic(self.model, num_samples=self.num_samples, mode="dropout")
