import torch
import torch.nn as nn

from bensemble.core.ensemble import Ensemble
from bensemble.core.member import ExplicitMembers, StochasticMembers
from bensemble.layers import BayesianLinear


def test_ensemble_custom_combiner_used():
    """Custom combiner replaces the default mean."""
    models = [nn.Linear(4, 2) for _ in range(3)]

    def combiner(preds):
        return preds.max(dim=0).values

    ensemble = Ensemble(ExplicitMembers(models), combiner=combiner)
    default_ensemble = Ensemble.from_models(models)
    x = torch.randn(8, 4)
    assert not torch.allclose(ensemble(x), default_ensemble(x))


def test_ensemble_custom_combiner_output_shape():
    """Custom combiner produces the expected output shape."""
    models = [nn.Linear(4, 2) for _ in range(3)]
    ensemble = Ensemble(ExplicitMembers(models), combiner=lambda p: p.sum(dim=0))
    assert ensemble(torch.randn(8, 4)).shape == (8, 2)


def test_stochastic_members_detects_bayesian():
    """Auto mode detects Bayesian layers and sets mode to 'bayesian'."""
    sm = StochasticMembers(nn.Sequential(BayesianLinear(4, 2)), mode="auto")
    assert sm.mode == "bayesian"


def test_stochastic_members_detects_dropout():
    """Auto mode detects Dropout layers and sets mode to 'dropout'."""
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))
    sm = StochasticMembers(model, mode="auto")
    assert sm.mode == "dropout"


def test_stochastic_members_detects_both():
    """Auto mode detects mixed Bayesian+Dropout layers and sets mode to 'both'."""
    model = nn.Sequential(BayesianLinear(4, 8), nn.Dropout(0.5))
    sm = StochasticMembers(model, mode="auto")
    assert sm.mode == "both"


def test_stochastic_members_fallback_to_dropout():
    """Plain model with no stochastic layers falls back to dropout mode."""
    sm = StochasticMembers(nn.Sequential(nn.Linear(4, 2)), mode="auto")
    assert sm.mode == "dropout"


def test_stochastic_members_produces_variance():
    """Dropout model in stochastic mode yields varied predictions across samples."""
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))
    sm = StochasticMembers(model, num_samples=20, mode="dropout")
    preds = sm.predict_all(torch.randn(4, 4))
    assert preds.var(dim=0).sum() > 0


def test_ensemble_on_device(device):
    """Ensemble runs correctly and output is on the given device."""
    models = [nn.Linear(4, 2).to(device) for _ in range(2)]
    ensemble = Ensemble.from_models(models)
    out = ensemble(torch.randn(8, 4, device=device))
    assert out.device.type == device.type
