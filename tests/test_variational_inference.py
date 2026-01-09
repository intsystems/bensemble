import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl

# TODO: move files like this
# from bensemble.layers import BayesianLinear
# from bensemble.likelihoods import GaussianLikelihood
# from bensemble.methods import VariationalEnsemble

from bensemble.methods.variational_inference import (
    VariationalEnsemble,
    BayesianLinear,
    GaussianLikelihood,
)

torch.manual_seed(42)


@pytest.fixture
def simple_data():
    """
    Генерирует данные для простой линейной регрессии.
    """
    torch.manual_seed(42)
    X = torch.randn(20, 5)
    y = torch.randn(20, 1)
    dataset = TensorDataset(X, y)

    loader = DataLoader(dataset, batch_size=5, num_workers=0)
    return X, y, loader


@pytest.fixture
def base_model():
    """
    Simple MLP model: Linear -> ReLU -> Linear
    """
    return nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))


@pytest.fixture
def nested_model():
    return nn.Sequential(nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 1)))


# --- COMPONENTS TESTS (Layers & Likelihood) ---


def test_bayesian_linear_shapes():
    layer = BayesianLinear(5, 3)
    x = torch.randn(10, 5)
    out = layer(x)
    assert out.shape == (10, 3)


def test_bayesian_linear_kl():
    """Checks that KL divergence returns scalar."""
    layer = BayesianLinear(5, 3)
    kl = layer.kl_divergence()
    assert isinstance(kl, torch.Tensor)
    assert kl.dim() == 0
    assert kl > -1e-5, "KL divergence cannot be negaitve."


def test_bayesian_linear_sampling_switch():
    layer = BayesianLinear(5, 3)
    x = torch.randn(10, 5)

    # 1. Режим сэмплирования (шум)
    layer.sampling = True
    torch.manual_seed(42)
    out1 = layer(x)
    torch.manual_seed(42)
    out2 = layer(x)
    assert torch.allclose(out1, out2), "Outputs must be the same."

    torch.manual_seed(43)
    out3 = layer(x)
    assert not torch.allclose(out1, out3), "Outputs must be different."

    layer.sampling = False
    out_det1 = layer(x)
    out_det2 = layer(x)
    assert torch.allclose(out_det1, out_det2), (
        "In non-sampling mode outputs must be the same."
    )


def test_gaussian_likelihood():
    lik = GaussianLikelihood(init_log_sigma=0.0)
    preds = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([[1.0], [2.0]])

    loss = lik(preds, target)
    assert isinstance(loss, torch.Tensor)

    sigma = lik.get_noise_sigma()
    assert isinstance(sigma, float)
    assert sigma > 0


# --- INTEGRATION TESTS (VariationalEnsemble + Lightning) ---


def test_initialization_and_conversion(base_model, simple_data):
    _, y, _ = simple_data
    ensemble = VariationalEnsemble(base_model, num_training_samples=len(y))

    bayesian_layers_count = 0
    for module in ensemble.model.modules():
        if isinstance(module, BayesianLinear):
            bayesian_layers_count += 1

    assert bayesian_layers_count == 2, "Must be 2 BayesianLinear layers"


def test_nested_conversion(nested_model, simple_data):
    _, y, _ = simple_data
    ensemble = VariationalEnsemble(nested_model, num_training_samples=len(y))

    bayesian_layers_count = sum(
        1 for m in ensemble.model.modules() if isinstance(m, BayesianLinear)
    )
    assert bayesian_layers_count == 2


def test_fit_process_lightning(base_model, simple_data):
    """
    Checks if model trains using Lightning Trainer without any errors.
    """
    X, y, loader = simple_data
    ensemble = VariationalEnsemble(base_model, num_training_samples=len(y))

    trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", logger=False)

    trainer.fit(ensemble, loader)


def test_predict(base_model, simple_data):
    X, y, loader = simple_data
    ensemble = VariationalEnsemble(base_model, num_training_samples=len(y))

    trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", logger=False)
    trainer.fit(ensemble, loader)

    mean, std = ensemble.predict(X, n_samples=5)

    assert mean.shape == y.shape
    assert std.shape == y.shape
    assert torch.all(std >= 0), "std cannot be negative."


def test_saving_loading_checkpoint(base_model, simple_data, tmp_path):
    """
    Checks saving and loading using Lightning Checkpoints.
    """
    X, y, loader = simple_data
    ensemble = VariationalEnsemble(base_model, num_training_samples=len(y))

    trainer = pl.Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=True,
    )
    trainer.fit(ensemble, loader)

    checkpoint_path = trainer.checkpoint_callback.best_model_path
    assert checkpoint_path, "Checkpoint must have been created."

    loaded_ensemble = VariationalEnsemble.load_from_checkpoint(
        checkpoint_path,
        model=base_model,
        num_training_samples=len(y),
        weights_only=False,
    )

    ensemble.eval()
    ensemble._set_sampling_mode(False)
    loaded_ensemble.eval()
    loaded_ensemble._set_sampling_mode(False)

    with torch.no_grad():
        orig_pred = ensemble(X)
        load_pred = loaded_ensemble(X)

    assert torch.allclose(orig_pred, load_pred), (
        "Loaded model must output the same result."
    )


# --- MATHEMATICAL CORRECTNESS TEST ---


def test_kl_formula_correctness():
    in_f, out_f = 4, 3
    prior_sigma = 1.5
    layer = BayesianLinear(in_f, out_f, prior_sigma=prior_sigma, init_sigma=0.2)

    with torch.no_grad():
        layer.w_mu.normal_(mean=0.1, std=0.2)
        layer.b_mu.normal_(mean=0.0, std=0.1)

    kl_method = layer.kl_divergence().detach()

    w_sigma = F.softplus(layer.w_rho)
    b_sigma = F.softplus(layer.b_rho)

    term1_w = torch.log(prior_sigma / w_sigma)
    term2_w = (w_sigma.pow(2) + layer.w_mu.pow(2)) / (2 * prior_sigma**2)
    kl_w = (term1_w + term2_w - 0.5).sum()

    term1_b = torch.log(prior_sigma / b_sigma)
    term2_b = (b_sigma.pow(2) + layer.b_mu.pow(2)) / (2 * prior_sigma**2)
    kl_b = (term1_b + term2_b - 0.5).sum()

    kl_manual = kl_w + kl_b

    assert torch.allclose(kl_method, kl_manual, rtol=1e-4)
