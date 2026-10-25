import pytest
import torch
from torch import nn

from bensemble.core.ensemble import Ensemble
from bensemble.diversity.dropout import MCDropoutEnsembler
from bensemble.layers import BayesianLinear
from bensemble.utils import enable_dropout


@pytest.fixture(
    params=[
        (nn.Dropout, (8, 4)),
        (nn.Dropout1d, (8, 4, 2)),
        (nn.Dropout2d, (8, 4, 2, 2)),
        (nn.Dropout3d, (8, 4, 2, 2, 2)),
        (nn.AlphaDropout, (8, 4)),
        (nn.FeatureAlphaDropout, (8, 4, 2, 2)),
    ],
    ids=lambda case: case[0].__name__,
)
def dropout_case(request):
    dropout_type, shape = request.param
    return dropout_type(p=0.5), torch.ones(shape)


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


@pytest.mark.parametrize("mode", ["auto", "dropout", "both"])
def test_stochastic_ensemble_samples_dropout_variants(dropout_case, mode):
    torch.manual_seed(0)
    dropout, x = dropout_case
    batch_norm = nn.BatchNorm1d(x[0].numel())
    model = nn.Sequential(dropout, nn.Flatten(), batch_norm).eval()
    ensemble = Ensemble.from_stochastic(model, num_samples=6, mode=mode)

    predictions = ensemble.predict_members(x)

    assert predictions.shape == (6, x.shape[0], x[0].numel())
    assert predictions.var(dim=0).sum() > 0
    assert batch_norm.num_batches_tracked.item() == 0
    assert all(not module.training for module in model.modules())


def test_auto_detects_bayesian_and_dropout_variants(dropout_case):
    dropout, x = dropout_case
    num_features = x[0].numel()
    model = nn.Sequential(
        nn.Flatten(),
        BayesianLinear(num_features, num_features),
        nn.Unflatten(1, x.shape[1:]),
        dropout,
    ).eval()

    ensemble = Ensemble.from_stochastic(model)

    assert ensemble.members.mode == "both"


def test_enable_dropout_variants_preserves_batch_norm(dropout_case):
    torch.manual_seed(0)
    dropout, x = dropout_case
    batch_norm = nn.BatchNorm1d(x[0].numel())
    model = nn.Sequential(dropout, nn.Flatten(), batch_norm).eval()

    enable_dropout(model)

    assert dropout.training
    assert not model.training
    assert not batch_norm.training
    assert not torch.equal(model(x), model(x))
    assert batch_norm.num_batches_tracked.item() == 0
