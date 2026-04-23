import torch.nn as nn

from bensemble.core.ensemble import Ensemble
from bensemble.diversity.dropout import MCDropoutEnsembler


def _make_model():
    return nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 2))


def test_mc_dropout_init():
    """MCDropoutEnsembler stores the wrapped model reference."""
    model = _make_model()
    ensembler = MCDropoutEnsembler(model)
    assert ensembler.model is model


def test_mc_dropout_build_returns_ensemble():
    """build_ensemble returns an Ensemble with the requested number of members."""
    model = _make_model()
    ensembler = MCDropoutEnsembler(model)
    result = ensembler.build_ensemble(num_samples=10)
    assert isinstance(result, Ensemble)
    assert result.num_members == 10
