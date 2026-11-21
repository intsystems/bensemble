from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bensemble.methods.laplace_approximation import LaplaceApproximation


def _weight_covariance(laplace):
    weight = laplace.model.weight.detach()
    basis = torch.eye(weight.numel(), dtype=weight.dtype).reshape(-1, *weight.shape)
    # Each unit noise vector gives one column of the sampling transform.
    with patch.object(torch, "randn", side_effect=basis.unbind()):
        samples = laplace.sample_models(len(basis))
    offsets = torch.stack(
        [(model.weight.detach() - weight).flatten() for model in samples]
    )
    return offsets.T @ offsets


def test_scalar_posterior_variance_uses_dataset_size_once():
    x = torch.ones(100, 1, dtype=torch.float64)
    loader = DataLoader(TensorDataset(x, torch.zeros_like(x)), batch_size=5)
    laplace = LaplaceApproximation(nn.Linear(1, 1, bias=False).double(), damping=0)
    laplace.compute_curvature(loader, num_samples=7)

    # Unit prior and observation variance give posterior precision 1 + sum(x**2).
    torch.testing.assert_close(_weight_covariance(laplace), x.new_tensor([[1 / 101]]))


def test_classification_covariance_matches_dense_posterior_hessian():
    x = torch.tensor([[1.0, 2.0], [-2.0, 0.5], [0.5, 1.5]], dtype=torch.float64)
    labels = torch.tensor([2, 1, 0])
    model = nn.Linear(2, 3).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(x.new_tensor([0.2, -0.5, 1.0]))
    prior_precision = 2.5
    laplace = LaplaceApproximation(
        model, likelihood="classification", prior_precision=prior_precision, damping=0
    )
    laplace.compute_curvature(DataLoader(TensorDataset(x, labels), batch_size=2))

    # Constant logits make K-FAC exact here. Adding the same weight vector to
    # every class leaves softmax unchanged, so only the prior acts in that subspace.
    def negative_log_posterior(flat_weight):
        logits = F.linear(x, flat_weight.view_as(model.weight), model.bias)
        return F.cross_entropy(logits, labels, reduction="sum") + (
            prior_precision / 2 * flat_weight.square().sum()
        )

    precision = torch.autograd.functional.hessian(
        negative_log_posterior, model.weight.flatten()
    )
    torch.testing.assert_close(_weight_covariance(laplace), torch.linalg.inv(precision))


@pytest.mark.parametrize(
    ("mode", "variance"), [("legacy", 1 / 49), ("paper", (3**0.5 + 2) ** -2)]
)
def test_previous_factor_modes_and_saved_samples(mode, variance):
    # N=3, x=1, prior precision=4: each historical mode has a known scalar variance.
    x = torch.ones(3, 1, dtype=torch.float64)
    model = nn.Linear(1, 1, bias=False).double()
    laplace = LaplaceApproximation(
        model, regularization=mode, prior_precision=4, damping=0
    )
    laplace.compute_curvature(DataLoader(TensorDataset(x, x), batch_size=3))
    state = laplace._get_ensemble_state()
    del state["regularization"], state["damping"]  # Older checkpoints omit these.
    restored = LaplaceApproximation(model)
    restored._set_ensemble_state(state)
    torch.testing.assert_close(_weight_covariance(restored), x.new_tensor([[variance]]))


def test_unregularized_softmax_has_no_finite_posterior_covariance():
    x = torch.ones(4, 1, dtype=torch.float64)
    model = nn.Linear(1, 3, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(x.new_tensor([[0.2], [-0.5], [1.0]]))
    laplace = LaplaceApproximation(
        model, likelihood="classification", prior_precision=0, damping=0
    )
    with pytest.raises(ValueError, match="Singular Laplace precision"):
        laplace.compute_curvature(DataLoader(TensorDataset(x, torch.zeros(4).long())))
