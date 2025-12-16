import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.distributions import Normal

from bensemble.methods.variational_inference import (
    BayesianLinear,
    GaussianLikelihood,
    VariationalEnsemble,
)


torch.manual_seed(42)


@pytest.fixture
def simple_data():
    """
    Generates simple linear regression data: y = 2x + 1 + noise.
    Returns: X, y, and a DataLoader.
    """
    torch.manual_seed(42)
    X = torch.randn(20, 5)
    # Random targets for basic execution testing
    y = torch.randn(20)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)
    return X, y, loader


@pytest.fixture
def base_model():
    """
    A simple flat MLP model for basic conversion tests.
    Structure: Linear -> ReLU -> Linear
    """
    return nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))


@pytest.fixture
def nested_model():
    """
    A nested model (Sequential inside Sequential) to test recursive layer replacement.
    """
    return nn.Sequential(nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 1)))


def test_bayesian_linear_shapes():
    """Checks if the output shape of the layer is correct."""
    layer = BayesianLinear(5, 3)
    x = torch.randn(10, 5)
    out = layer(x)
    assert out.shape == (10, 3)


def test_bayesian_linear_kl():
    """Checks if KL divergence calculation returns a scalar tensor."""
    layer = BayesianLinear(5, 3)
    kl = layer.kl_divergence()
    assert isinstance(kl, torch.Tensor)
    assert kl.dim() == 0
    assert kl > -1e-5


def test_bayesian_linear_deterministic_vs_stochastic():
    """
    Checks the behavior of sampling modes:
    1. With sampling=True, outputs should differ (due to noise).
    2. With sampling=False, outputs should be identical (deterministic).
    """
    layer = BayesianLinear(5, 3)
    x = torch.randn(10, 5)

    layer.sampling = True
    torch.manual_seed(42)
    out1 = layer(x)
    torch.manual_seed(42)
    out2 = layer(x)

    assert torch.allclose(out1, out2)

    out3 = layer(x)
    assert not torch.allclose(out1, out3)

    layer.sampling = False
    out_det1 = layer(x)
    out_det2 = layer(x)

    assert torch.allclose(out_det1, out_det2)


def test_gaussian_likelihood():
    """Checks Gaussian Likelihood loss calculation."""
    lik = GaussianLikelihood(init_log_sigma=0.0)
    preds = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 2.0])

    loss = lik(preds, target)
    assert isinstance(loss, torch.Tensor)

    sigma = lik.get_noise_sigma()
    assert isinstance(sigma, float)
    assert sigma > 0


def test_initialization_and_conversion(base_model):
    """Ensures that standard Linear layers are converted to BayesianLinear."""
    ensemble = VariationalEnsemble(base_model, auto_convert=True)

    is_bayesian = False
    for module in ensemble.model.modules():
        if isinstance(module, BayesianLinear):
            is_bayesian = True
            break
    assert is_bayesian, "Linear layers were not converted to BayesianLinear."


def test_nested_conversion(nested_model):
    """Ensures recursive conversion works for nested structures."""
    ensemble = VariationalEnsemble(nested_model, auto_convert=True)

    count_bayesian = 0
    for module in ensemble.model.modules():
        if isinstance(module, BayesianLinear):
            count_bayesian += 1

    assert count_bayesian == 2


def test_fit_process(simple_data, base_model):
    """Tests the training loop (Happy Path)."""
    X, y, loader = simple_data
    ensemble = VariationalEnsemble(base_model)

    history = ensemble.fit(loader, epochs=1, verbose=False)

    assert ensemble.is_fitted
    assert "train_loss" in history
    assert len(history["train_loss"]) == 1


def test_predict(simple_data, base_model):
    """Tests prediction functionality and output shapes."""
    X, y, loader = simple_data
    ensemble = VariationalEnsemble(base_model)
    ensemble.fit(loader, epochs=1, verbose=False)

    mean, std = ensemble.predict(X, n_samples=5)

    assert mean.squeeze().shape == y.shape
    assert std.squeeze().shape == y.shape
    assert torch.all(std >= 0), "Standard deviation cannot be negative."


def test_sample_models(simple_data, base_model):
    """Tests sampling of specific model instances (frozen weights) from the ensemble."""
    X, y, loader = simple_data
    ensemble = VariationalEnsemble(base_model)
    ensemble.fit(loader, epochs=1, verbose=False)

    models = ensemble.sample_models(n_models=3)
    assert len(models) == 3

    single_model = models[0]
    out1 = single_model(X)
    out2 = single_model(X)
    assert torch.allclose(out1, out2), "Sampled models must be deterministic."


def test_errors_not_fitted(base_model):
    """Ensures methods raise RuntimeError if called before fitting."""
    ensemble = VariationalEnsemble(base_model)
    ensemble.is_fitted = False

    x = torch.randn(5, 5)

    with pytest.raises(RuntimeError, match="not fitted"):
        ensemble.predict(x)

    with pytest.raises(RuntimeError, match="not fitted"):
        ensemble.sample_models()


def test_state_save_load(base_model, simple_data):
    """
    Tests that the model state can be saved and restored correctly.
    Verifies that a restored model produces identical predictions to the original.
    """
    X, y, loader = simple_data

    model_1 = VariationalEnsemble(base_model, auto_convert=True)
    model_1.fit(loader, epochs=1, verbose=False)

    state = model_1._get_ensemble_state()

    assert "model_state_dict" in state
    assert "likelihood_state_dict" in state
    assert state["is_fitted"] is True

    import copy

    fresh_base_model = copy.deepcopy(base_model)
    model_2 = VariationalEnsemble(fresh_base_model, auto_convert=True)

    assert not model_2.is_fitted

    model_2._set_ensemble_state(state)

    assert model_2.is_fitted is True
    assert model_2.prior_sigma == model_1.prior_sigma

    for p1, p2 in zip(model_1.model.parameters(), model_2.model.parameters()):
        assert torch.equal(p1, p2)

    assert model_1.likelihood.get_noise_sigma() == model_2.likelihood.get_noise_sigma()

    model_1.model.eval()
    model_2.model.eval()
    model_1._set_sampling_mode(False)
    model_2._set_sampling_mode(False)

    with torch.no_grad():
        pred_1 = model_1.model(X)
        pred_2 = model_2.model(X)

    assert torch.allclose(pred_1, pred_2), (
        "Restored model predictions do not match original"
    )


def test_optimizer_state_ignored_if_none(base_model):
    """
    Ensures that loading state into a model without an optimizer
    doesn't crash (optimizer state is simply ignored until fit is called).
    """
    model = VariationalEnsemble(base_model)

    fake_state = {
        "model_state_dict": model.model.state_dict(),
        "likelihood_state_dict": model.likelihood.state_dict(),
        "is_fitted": True,
        "prior_sigma": 1.0,
        "learning_rate": 0.01,
        "optimizer_state_dict": {"some": "garbage"},
    }

    model._set_ensemble_state(fake_state)
    assert model.is_fitted is True


def test_kl_matches_closed_form():
    """KL(q || p) computed by the layer should match manual closed-form formula."""
    in_f, out_f = 4, 3
    prior_sigma = 1.5
    layer = BayesianLinear(in_f, out_f, prior_sigma=prior_sigma, init_sigma=0.2)

    with torch.no_grad():
        layer.w_mu.normal_(mean=0.1, std=0.2)
        layer.b_mu.normal_(mean=0.0, std=0.1)

    kl_method = layer.kl_divergence().detach()

    w_mu = layer.w_mu
    b_mu = layer.b_mu
    w_sigma = F.softplus(layer.w_rho)
    b_sigma = F.softplus(layer.b_rho)
    var_q_w = w_sigma.pow(2)
    var_q_b = b_sigma.pow(2)
    var_p = prior_sigma**2

    kl_w = (
        torch.log(prior_sigma / w_sigma) + (var_q_w + w_mu.pow(2)) / (2.0 * var_p) - 0.5
    ).sum()
    kl_b = (
        torch.log(prior_sigma / b_sigma) + (var_q_b + b_mu.pow(2)) / (2.0 * var_p) - 0.5
    ).sum()
    kl_manual = kl_w + kl_b

    assert torch.allclose(kl_method, kl_manual, rtol=1e-4, atol=1e-6)


def test_gaussian_likelihood_matches_normal_logprob():
    """GaussianLikelihood should equal negative sum of Normal(preds, sigma).log_prob(target)."""
    lik = GaussianLikelihood(init_log_sigma=-0.5)

    for preds, target in [
        (torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])),
        (torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [2.0]])),
    ]:
        preds = preds.float()
        target = target.float()
        loss = lik(preds, target)
        sigma = F.softplus(lik.log_sigma) + 1e-3  # тензор shape [1]
        var = sigma**2
        expected = 0.5 * (torch.log(var) + (preds - target) ** 2 / var).sum()

        assert torch.allclose(loss, expected, rtol=1e-6, atol=1e-6)


def test_local_reparam_empirical_variance():
    """
    Checks local reparameterization trick.
    """
    torch.manual_seed(123)
    in_f, out_f = 6, 4
    batch = 8
    layer = BayesianLinear(in_f, out_f, init_sigma=0.3)

    with torch.no_grad():
        layer.w_mu.normal_(mean=0.0, std=0.1)
        layer.b_mu.normal_(mean=0.0, std=0.05)

    x = torch.randn(batch, in_f)

    w_sigma = F.softplus(layer.w_rho)
    b_sigma = F.softplus(layer.b_rho)
    delta = F.linear(x.pow(2), w_sigma.pow(2)) + b_sigma.pow(2)

    layer.sampling = True
    n_samples = 500
    samples = []
    for _ in range(n_samples):
        samples.append(layer(x).unsqueeze(0))
    all_samples = torch.cat(samples, dim=0)

    emp_var = all_samples.var(dim=0, unbiased=False)

    assert torch.allclose(emp_var, delta, rtol=0.18, atol=1e-2)
