from torch.utils.data import DataLoader

from bensemble.core.ensemble import Ensemble


class NESBayesianSampler:
    def __init__(self):
        pass

    def sample_mc(self, val_loader: DataLoader) -> Ensemble:
        """
        Args:
            val_loader (DataLoader): Used to evaluate the posterior.

        Returns:
            Ensemble: The final ensemble wrapped in bensemble's core abstraction.
        """
        pass
        # models = [model1, model2, ...]
        # return Ensemble.from_models(models)

    def sample_svgd(self, val_loader: DataLoader) -> Ensemble:
        """
        Args:
            val_loader (DataLoader): Used to evaluate the architecture's loss/posterior.

        Returns:
            Ensemble: The final ensemble wrapped in bensemble's core abstraction.
        """
        pass
        # models = [model1, model2, ...]
        # return Ensemble.from_models(models)
