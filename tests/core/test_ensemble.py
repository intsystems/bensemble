import torch
from torch import nn

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


def test_stochastic_members_both_mode_samples_bayesian_layers():
    """
    In 'both' mode the Bayesian layers must stay stochastic. With dropout at
    p=0 they are the only source of variance, so frozen weights give exactly 0.
    """
    torch.manual_seed(0)
    model = nn.Sequential(BayesianLinear(4, 8), nn.Dropout(0.0), nn.Linear(8, 2))
    x = torch.randn(4, 4)

    both = StochasticMembers(model, num_samples=10, mode="both").predict_all(x)
    bayesian = StochasticMembers(model, num_samples=10, mode="bayesian").predict_all(x)

    assert bayesian.var(dim=0).sum() > 0
    assert both.var(dim=0).sum() > 0


def test_stochastic_members_both_mode_activates_every_layer_kind():
    """After activation in 'both' mode, Bayesian and Dropout layers are both in train mode."""
    model = nn.Sequential(BayesianLinear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))
    sm = StochasticMembers(model, mode="both")
    model.eval()

    sm._activate()

    assert model[0].training
    assert model[1].training
    assert not model[2].training


def test_ensemble_on_device(device):
    """Ensemble runs correctly and output is on the given device."""
    models = [nn.Linear(4, 2).to(device) for _ in range(2)]
    ensemble = Ensemble.from_models(models)
    out = ensemble(torch.randn(8, 4, device=device))
    assert out.device.type == device.type
