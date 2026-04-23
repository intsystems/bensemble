import math

import torch
import torch.nn as nn

from bensemble.layers import BayesianLinear
from bensemble.utils import (
    enable_dropout,
    get_total_kl,
    predict_with_uncertainty,
    standard_normal_cdf,
    standard_normal_pdf,
)


def test_enable_dropout():
    """enable_dropout sets only Dropout layers to train mode, leaving others in eval."""
    model = nn.Sequential(nn.Linear(10, 10), nn.Dropout(0.5), nn.Linear(10, 1))

    model.eval()
    assert not model[1].training
    assert not model[0].training

    enable_dropout(model)

    assert model[1].training is True, "Dropout layer must be in train mode"
    assert model[0].training is False, "Linear layer should remain in eval mode"

    x = torch.ones(100, 10)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2), "Dropout must produce different outputs across forward passes"


def test_standard_normal_pdf_at_zero():
    """PDF at 0 matches the analytical value 1/sqrt(2π)."""
    x = torch.tensor([0.0])
    result = standard_normal_pdf(x)
    assert abs(result.item() - 1.0 / math.sqrt(2 * math.pi)) < 1e-4


def test_standard_normal_pdf_symmetry():
    """PDF is symmetric: pdf(x) == pdf(-x)."""
    x_pos = torch.tensor([1.0])
    x_neg = torch.tensor([-1.0])
    assert torch.allclose(standard_normal_pdf(x_pos), standard_normal_pdf(x_neg))


def test_standard_normal_cdf_at_zero():
    """CDF at 0 equals 0.5."""
    x = torch.tensor([0.0])
    assert abs(standard_normal_cdf(x).item() - 0.5) < 1e-6


def test_standard_normal_cdf_bounds():
    """CDF approaches 0 at -∞ and 1 at +∞."""
    x_low = torch.tensor([-10.0])
    x_high = torch.tensor([10.0])
    assert standard_normal_cdf(x_low).item() < 1e-4
    assert standard_normal_cdf(x_high).item() > 0.9999


def test_get_total_kl_positive():
    """Total KL is positive for a model with Bayesian layers."""
    model = nn.Sequential(BayesianLinear(4, 4), BayesianLinear(4, 2))
    kl = get_total_kl(model)
    assert kl > 0


def test_get_total_kl_no_bayesian_layers():
    """Total KL is zero for a plain model with no Bayesian layers."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    kl = get_total_kl(model)
    assert kl == 0.0


def test_predict_with_uncertainty_shapes():
    """Mean and std outputs have the expected shape and std is non-negative."""
    model = nn.Sequential(BayesianLinear(4, 8), nn.ReLU(), BayesianLinear(8, 1))
    x = torch.randn(8, 4)
    mean, std = predict_with_uncertainty(model, x, num_samples=5)
    assert mean.shape == (8, 1)
    assert std.shape == (8, 1)
    assert (std >= 0).all()


def test_predict_with_uncertainty_restores_mode():
    """Model training mode is restored to its original state after prediction."""
    model = nn.Sequential(BayesianLinear(4, 4), BayesianLinear(4, 1))
    model.train()
    x = torch.randn(4, 4)
    predict_with_uncertainty(model, x, num_samples=3)
    assert model.training is True
