import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from bensemble.methods.laplace_approximation import LaplaceApproximation


@pytest.fixture
def regression_setup():
    """
    Sets up a simple regression problem.
    Data: X (20, 5), y (20, 1)
    Model: MLP returning (N, 1)
    """
    torch.manual_seed(42)
    X = torch.randn(20, 5)
    y = (2 * X.sum(dim=1, keepdim=True)) + torch.randn(20, 1) * 0.1

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)

    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
    return model, loader, X, y


@pytest.fixture
def classification_setup():
    """
    Sets up a simple classification problem.
    Data: X (20, 5), y (20,) with 3 classes
    Model: MLP returning logits (N, 3)
    """
    torch.manual_seed(42)
    X = torch.randn(20, 5)
    y = torch.randint(0, 3, (20,))

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)

    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
    )
    return model, loader, X, y


def test_initialization_errors(regression_setup):
    """
    Tests initialization validation.
    Ensures ValueError is raised for unsupported likelihoods.
    """
    model, _, _, _ = regression_setup

    laplace = LaplaceApproximation(model, likelihood="regression")
    assert laplace.likelihood == "regression"

    with pytest.raises(ValueError, match="Unsupported likelihood"):
        LaplaceApproximation(model, likelihood="magic_likelihood")


def test_verbose_toggle(regression_setup):
    """Tests the verbose toggle functionality."""
    model, _, _, _ = regression_setup
    laplace = LaplaceApproximation(model)

    laplace.verbose = False

    laplace.toggle_verbose()
    assert laplace.verbose is True

    laplace.toggle_verbose()
    assert laplace.verbose is False


def test_fit_regression_no_pretrained(regression_setup):
    """
    Tests full pipeline for regression (MSE):
    1. Training from scratch (pretrained=False).
    2. Computing Laplace factors (K-FAC).
    """
    model, loader, _, _ = regression_setup

    laplace = LaplaceApproximation(
        model, pretrained=False, likelihood="regression", verbose=True
    )

    history = laplace.fit(loader, num_epochs=1, num_samples=10)

    assert "train_loss" in history
    assert len(history["train_loss"]) == 1
    assert laplace.pretrained is True

    assert len(laplace.kronecker_factors) == 2
    assert len(laplace.sampling_factors) == 2

    for name, factors in laplace.sampling_factors.items():
        assert "L_U" in factors
        assert "L_V" in factors
        assert "weight_shape" in factors


def test_fit_classification_pretrained(classification_setup):
    """
    Tests pipeline for classification (CrossEntropy):
    1. Skips MAP training (pretrained=True).
    2. Computes factors using CrossEntropy Hessian approximation.
    """
    model, loader, _, _ = classification_setup

    laplace = LaplaceApproximation(
        model, pretrained=True, likelihood="classification", verbose=False
    )

    laplace.fit(loader, num_samples=10)
    assert len(laplace.kronecker_factors) > 0

    for name, factors in laplace.kronecker_factors.items():
        assert not torch.isnan(factors["Q"]).any()
        assert not torch.isnan(factors["H"]).any()


def test_predict_regression_shapes(regression_setup):
    """
    Tests prediction output shapes and values for regression.
    """
    model, loader, X, y = regression_setup
    laplace = LaplaceApproximation(model, likelihood="regression")
    laplace.fit(loader, num_epochs=0, num_samples=10)  # Fast fit

    mean, var = laplace.predict(X, n_samples=5)

    assert mean.shape == y.shape
    assert var.shape == y.shape

    assert (var >= 0).all()


def test_predict_classification_shapes(classification_setup):
    """
    Tests prediction output shapes for classification.
    Expects probabilities and entropy/uncertainty.
    """
    model, loader, X, _ = classification_setup
    laplace = LaplaceApproximation(model, likelihood="classification")
    laplace.fit(loader, num_epochs=0, num_samples=10)

    probs, uncertainty = laplace.predict(X, n_samples=5)

    assert probs.shape == (20, 3)

    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums))

    assert uncertainty.shape == (20,)
    assert (uncertainty >= 0).all()


def test_sample_models_diversity(regression_setup):
    """
    Ensures that sampled models are:
    1. Valid nn.Modules.
    2. Have different weights (stochasticity).
    """
    model, loader, _, _ = regression_setup
    laplace = LaplaceApproximation(model, likelihood="regression")
    laplace.fit(loader, num_epochs=0, num_samples=10)

    samples = laplace.sample_models(n_models=2)
    assert len(samples) == 2

    original_w = list(model.parameters())[0]
    sample1_w = list(samples[0].parameters())[0]
    sample2_w = list(samples[1].parameters())[0]

    assert not torch.equal(original_w, sample1_w)
    assert not torch.equal(sample1_w, sample2_w)


def test_hooks_cleanup(regression_setup):
    """
    Ensures that PyTorch forward hooks are removed after fitting.
    Leaving hooks can cause memory leaks or unexpected behavior.
    """
    model, loader, _, _ = regression_setup
    laplace = LaplaceApproximation(model, likelihood="regression")

    assert len(laplace.hook_handles) == 0

    laplace.fit(loader, num_samples=5)

    assert len(laplace.hook_handles) == 0


def test_state_management(regression_setup):
    """
    Tests _get_ensemble_state and _set_ensemble_state.
    Assumes bugs with empty keys have been fixed.
    """
    model, loader, _, _ = regression_setup
    laplace = LaplaceApproximation(model, likelihood="regression")
    laplace.fit(loader, num_epochs=0, num_samples=5)

    state = laplace._get_ensemble_state()

    assert state["likelihood"] == "regression"

    required_keys = ["kronecker_factors", "sampling_factors", "dataset_size"]
    for key in required_keys:
        assert key in state

    new_laplace = LaplaceApproximation(model, likelihood="regression")

    new_laplace._set_ensemble_state(state)

    assert new_laplace.dataset_size == laplace.dataset_size
    assert len(new_laplace.sampling_factors) == len(laplace.sampling_factors)
