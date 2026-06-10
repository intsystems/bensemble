import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from bensemble.methods.laplace_approximation import LaplaceApproximation
from bensemble.core.ensemble import Ensemble


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


def test_compute_curvature_regression(regression_setup):
    """
    Tests full pipeline for regression:
    """
    model, loader, _, _ = regression_setup

    laplace = LaplaceApproximation(model, likelihood="regression", verbose=True)

    # Use the new method name
    laplace.compute_curvature(loader, num_samples=10)

    assert laplace.is_fitted is True
    assert len(laplace.kronecker_factors) == 2
    assert len(laplace.sampling_factors) == 2

    for name, factors in laplace.sampling_factors.items():
        assert "L_U" in factors
        assert "L_V" in factors
        assert "weight_shape" in factors


def test_compute_curvature_classification(classification_setup):
    """
    Tests pipeline for classification:
    """
    model, loader, _, _ = classification_setup

    laplace = LaplaceApproximation(model, likelihood="classification", verbose=False)

    laplace.compute_curvature(loader, num_samples=10)
    assert len(laplace.kronecker_factors) > 0

    for name, factors in laplace.kronecker_factors.items():
        assert not torch.isnan(factors["Q"]).any()
        assert not torch.isnan(factors["H"]).any()


def test_ensemble_integration_regression(regression_setup):
    """
    Tests integration for regression.
    """
    model, loader, X, y = regression_setup
    laplace = LaplaceApproximation(model, likelihood="regression")
    laplace.compute_curvature(loader, num_samples=10)

    # Use the new API
    ensemble = Ensemble.from_posterior(laplace, n_members=5)

    # Predict members directly
    with torch.no_grad():
        member_preds = ensemble.predict_members(X)

    assert member_preds.shape == (5, 20, 1)


def test_ensemble_integration_classification(classification_setup):
    """
    Tests integration for classification.
    """
    model, loader, X, _ = classification_setup
    laplace = LaplaceApproximation(model, likelihood="classification")
    laplace.compute_curvature(loader, num_samples=10)

    ensemble = Ensemble.from_posterior(laplace, n_members=5)

    with torch.no_grad():
        member_preds = ensemble.predict_members(X)

    assert member_preds.shape == (5, 20, 3)


def test_sample_models_diversity(regression_setup):
    """
    Ensures that sampled models are:
    1. Valid nn.Modules.
    2. Have different weights.
    """
    model, loader, _, _ = regression_setup
    laplace = LaplaceApproximation(model, likelihood="regression")
    laplace.compute_curvature(loader, num_samples=10)

    samples = laplace.sample_models(n_models=2)
    assert len(samples) == 2

    original_w = list(model.parameters())[0]
    sample1_w = list(samples[0].parameters())[0]
    sample2_w = list(samples[1].parameters())[0]

    assert not torch.equal(original_w, sample1_w)
    assert not torch.equal(sample1_w, sample2_w)


def test_hooks_cleanup(regression_setup):
    """
    Ensures that PyTorch forward hooks are removed after computing curvature.
    """
    model, loader, _, _ = regression_setup
    laplace = LaplaceApproximation(model, likelihood="regression")

    assert len(laplace.hook_handles) == 0

    laplace.compute_curvature(loader, num_samples=5)

    assert len(laplace.hook_handles) == 0


def test_state_management(regression_setup):
    """
    Tests _get_ensemble_state and _set_ensemble_state.
    """
    model, loader, _, _ = regression_setup
    laplace = LaplaceApproximation(model, likelihood="regression")
    laplace.compute_curvature(loader, num_samples=5)

    state = laplace._get_ensemble_state()

    assert state["likelihood"] == "regression"

    required_keys = ["kronecker_factors", "sampling_factors", "dataset_size"]
    for key in required_keys:
        assert key in state

    new_laplace = LaplaceApproximation(model, likelihood="regression")
    new_laplace._set_ensemble_state(state)

    assert new_laplace.dataset_size == laplace.dataset_size
    assert len(new_laplace.sampling_factors) == len(laplace.sampling_factors)
