import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


from bensemble.methods.probabilistic_backpropagation import (
    ProbabilisticBackpropagation,
    PBPNet,
    ProbLinear,
    phi,
    Phi,
    relu_moments,
)


@pytest.fixture
def pbp_data():
    """
    Generates regression data for PBP testing.
    PBP usually works with double precision (float64).
    """
    torch.manual_seed(42)
    X = torch.randn(20, 1, dtype=torch.float64)
    y = X**3 + torch.randn(20, 1, dtype=torch.float64) * 0.1

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)
    return X, y, loader


@pytest.fixture
def pbp_model_setup():
    """Returns a simple PBP model initialized with layer sizes."""
    return [1, 10, 1]


def test_math_helpers():
    """Tests phi (PDF) and Phi (CDF) functions."""
    x = torch.tensor([0.0], dtype=torch.float64)

    assert torch.abs(phi(x) - 0.3989) < 1e-3

    assert torch.abs(Phi(x) - 0.5) < 1e-3


def test_relu_moments():
    """Tests analytical moment propagation through ReLU."""
    mean = torch.tensor([0.0], dtype=torch.float64)
    var = torch.tensor([1.0], dtype=torch.float64)

    m_out, v_out = relu_moments(mean, var)

    assert torch.abs(m_out - 0.3989) < 1e-3
    assert v_out > 0


def test_prob_linear_init():
    """Tests initialization of ProbLinear layer."""
    layer = ProbLinear(5, 3, dtype=torch.float64)

    assert layer.m.shape == (3, 6)
    assert layer.v.shape == (3, 6)
    assert layer.m.dtype == torch.float64


def test_pbp_net_forward():
    """Tests forward pass of moments in PBPNet."""
    net = PBPNet([2, 5, 1], dtype=torch.float64)
    x = torch.randn(10, 2, dtype=torch.float64)

    mz, vz = net.forward_moments(x)

    assert mz.shape == (10, 1)
    assert vz.shape == (10, 1)
    assert (vz >= 0).all()


def test_initialization(pbp_model_setup):
    """Tests initialization of the PBP wrapper."""
    pbp = ProbabilisticBackpropagation(layer_sizes=pbp_model_setup)
    assert isinstance(pbp.model, PBPNet)

    with pytest.raises(ValueError, match="Specify either"):
        ProbabilisticBackpropagation(model=None, layer_sizes=None)


def test_fit_loop(pbp_data, pbp_model_setup):
    """
    Tests the main training loop (ADF updates).
    Checks if parameters update and history is returned.
    """
    X, y, loader = pbp_data
    pbp = ProbabilisticBackpropagation(layer_sizes=pbp_model_setup, dtype=torch.float64)

    alpha_old = pbp.alpha_g.item()
    beta_old = pbp.beta_g.item()

    history = pbp.fit(loader, num_epochs=1, prior_refresh=0)

    assert pbp.is_fitted
    assert "train_rmse" in history

    assert pbp.alpha_g.item() != alpha_old or pbp.beta_g.item() != beta_old


def test_prior_refresh(pbp_data, pbp_model_setup):
    """Tests the prior refresh mechanism (updating alpha_l, beta_l)."""
    _, _, loader = pbp_data
    pbp = ProbabilisticBackpropagation(layer_sizes=pbp_model_setup, dtype=torch.float64)
    pbp.fit(loader, num_epochs=1, prior_refresh=1)

    assert isinstance(pbp.alpha_l, torch.Tensor)


def test_predict(pbp_data, pbp_model_setup):
    """Tests prediction output (mean and samples)."""
    X, y, loader = pbp_data
    pbp = ProbabilisticBackpropagation(layer_sizes=pbp_model_setup, dtype=torch.float64)
    pbp.fit(loader, num_epochs=1)

    mean, samples = pbp.predict(X, n_samples=5)

    assert mean.shape == y.shape

    assert samples.shape == (5,) + y.shape

    noise_var = pbp.noise_variance()
    assert noise_var > 0


def test_sample_models(pbp_data, pbp_model_setup):
    """Tests sampling of PyTorch models from PBP posterior."""
    X, y, loader = pbp_data
    pbp = ProbabilisticBackpropagation(layer_sizes=pbp_model_setup, dtype=torch.float64)
    pbp.fit(loader, num_epochs=1)

    models = pbp.sample_models(n_models=2)
    assert len(models) == 2
    assert isinstance(models[0], nn.Sequential)

    model = models[0]

    out = model(X)
    assert out.shape == y.shape


def test_state_management(pbp_model_setup):
    """Tests saving and loading ensemble state."""
    pbp = ProbabilisticBackpropagation(layer_sizes=pbp_model_setup, dtype=torch.float64)

    state = pbp._get_ensemble_state()

    assert "alpha_g" in state
    assert "beta_l" in state

    pbp.alpha_g = torch.tensor(100.0, dtype=torch.float64)
    pbp._set_ensemble_state(state)

    assert pbp.alpha_g.item() == 6.0


def test_val_loader(pbp_data, pbp_model_setup):
    """Tests fit with validation loader."""
    X, y, loader = pbp_data
    pbp = ProbabilisticBackpropagation(layer_sizes=pbp_model_setup, dtype=torch.float64)

    history = pbp.fit(loader, val_loader=loader, num_epochs=1)

    assert "val_rmse" in history
    assert len(history["val_rmse"]) == 1


def test_predict_not_fitted(pbp_model_setup):
    """Ensures predict raises error if not fitted."""
    pbp = ProbabilisticBackpropagation(layer_sizes=pbp_model_setup)
    X = torch.randn(5, 1)
    with pytest.raises(RuntimeError, match="Model not fitted"):
        pbp.predict(X)
