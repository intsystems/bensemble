import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from bensemble import VariationalRenyi


@pytest.fixture
def regression_setup():
    """
    Sets up a simple regression environment.
    Model: Linear(5, 1) -> Regression path in _compute_log_weights.
    """
    torch.manual_seed(42)
    X = torch.randn(20, 5)
    y = torch.randn(20, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)

    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
    return model, loader, X, y


@pytest.fixture
def classification_setup():
    """
    Sets up a simple classification environment.
    Model: Linear(5, 2) -> Classification path in _compute_log_weights.
    """
    torch.manual_seed(42)
    X = torch.randn(20, 5)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)

    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 2))
    return model, loader, X, y


def test_initialization_and_conversion(regression_setup):
    """
    Tests if the model is correctly converted to a Bayesian version.
    Checks for the existence of _mu and _rho parameters.
    """
    model, _, _, _ = regression_setup
    vr = VariationalRenyi(model, initial_rho=-3.0)

    has_bayesian_layer = False
    for module in vr.bayesian_model.modules():
        if isinstance(module, nn.Linear):
            assert hasattr(module, "weight_mu")
            assert hasattr(module, "weight_rho")
            assert isinstance(module.weight_mu, nn.Parameter)
            assert torch.all(module.weight_rho == -3.0)

            has_bayesian_layer = True

    assert has_bayesian_layer


def test_stochastic_forward(regression_setup):
    """
    Ensures that the forward pass is stochastic due to reparameterization.
    """
    model, _, X, _ = regression_setup
    vr = VariationalRenyi(model)

    torch.manual_seed(42)
    out1 = vr.forward(X)

    torch.manual_seed(43)
    out2 = vr.forward(X)

    assert not torch.allclose(out1, out2)


def test_fit_regression_alpha_1(regression_setup):
    """
    Tests the training loop for Regression with Alpha = 1.0.
    This triggers the standard VI loss path (ELBO).
    """
    model, loader, _, _ = regression_setup
    vr = VariationalRenyi(model, alpha=1.0)

    history = vr.fit(loader, num_epochs=1, n_samples=2)

    assert vr.is_fitted
    assert "train_loss" in history
    assert len(history["train_loss"]) == 1


def test_fit_classification_alpha_renyi(classification_setup):
    """
    Tests the training loop for Classification with Alpha != 1.0.
    This triggers the Rényi Bound loss path and CrossEntropy logic.
    """
    model, loader, _, _ = classification_setup
    vr = VariationalRenyi(model, alpha=0.5)

    history = vr.fit(loader, num_epochs=1, n_samples=2)

    assert vr.is_fitted
    assert "train_loss" in history


def test_gradient_clipping(regression_setup):
    """
    Tests if gradient clipping parameter is accepted and runs without errors.
    """
    model, loader, _, _ = regression_setup
    vr = VariationalRenyi(model)

    vr.fit(loader, num_epochs=1, grad_clip=1.0)


def test_validation_loop(regression_setup):
    """
    Tests the validation logic within fit.
    """
    model, loader, _, _ = regression_setup
    vr = VariationalRenyi(model)

    history = vr.fit(loader, val_loader=loader, num_epochs=1)

    assert "val_loss" in history
    assert len(history["val_loss"]) == 1


def test_predict(regression_setup):
    """
    Tests prediction output shapes.
    Returns: (mean_pred, all_samples)
    """
    model, loader, X, y = regression_setup
    vr = VariationalRenyi(model)
    vr.fit(loader, num_epochs=1)

    n_samples = 5
    mean_pred, raw_samples = vr.predict(X, n_samples=n_samples)

    assert mean_pred.shape == y.shape

    assert raw_samples.shape == (n_samples, *y.shape)


def test_sample_models(regression_setup):
    """
    Tests the model sampling functionality.
    Ensures that sampled models are valid and deterministic snapshots.
    """
    model, loader, X, _ = regression_setup
    vr = VariationalRenyi(model)
    vr.fit(loader, num_epochs=1)

    models = vr.sample_models(n_models=2)
    assert len(models) == 2

    sample = models[0]
    out1 = sample(X)
    out2 = sample(X)
    assert torch.allclose(out1, out2), "Sampled model should be deterministic"


def test_state_management(regression_setup):
    """
    Tests getting and setting the ensemble state.
    """
    model, loader, _, _ = regression_setup
    vr = VariationalRenyi(model, alpha=0.5)
    vr.fit(loader, num_epochs=1)

    state = vr._get_ensemble_state()
    assert state["alpha"] == 0.5
    assert "optimizer_state" in state

    vr_new = VariationalRenyi(model, alpha=1.0)
    vr_new._set_ensemble_state(state)

    assert vr_new.alpha == 0.5


def test_error_not_fitted(regression_setup):
    """
    Ensures predict raises RuntimeError if called before fit.
    """
    model, _, X, _ = regression_setup
    vr = VariationalRenyi(model)

    with pytest.raises(RuntimeError, match="Model not fitted"):
        vr.predict(X)
