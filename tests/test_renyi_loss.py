import math

import pytest
import torch
from torch import nn

from bensemble.losses import VariationalLoss


class ZeroLikelihood(nn.Module):
    def forward(self, preds, target):
        return preds * 0


class SquaredErrorLikelihood(nn.Module):
    def forward(self, preds, target):
        return (preds - target).square()


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_gaussian_renyi_divergence_matches_analytic_value(alpha):
    # D_alpha(N(1, 1) || N(0, 1)) = alpha / 2. Midpoint quantiles
    # integrate over q deterministically, without a random convergence test.
    mean = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    q = torch.distributions.Normal(mean, 1.0)
    p = torch.distributions.Normal(torch.tensor(0.0).double(), 1.0)
    quantiles = (torch.arange(16384, dtype=torch.float64) + 0.5) / 16384
    samples = q.icdf(quantiles)
    preds = torch.zeros(len(samples), 1, 1, dtype=torch.float64)

    loss = VariationalLoss(ZeroLikelihood(), alpha=alpha)(
        preds,
        torch.zeros(1, 1, dtype=torch.float64),
        log_prior=p.log_prob(samples),
        log_posterior=q.log_prob(samples),
    )

    assert loss.item() == pytest.approx(alpha / 2, abs=1e-3)
    loss.backward()
    assert mean.grad.item() == pytest.approx(alpha, abs=2e-3)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("num_batches", [1, 5])
def test_renyi_matches_finite_sample_joint_probability(alpha, num_batches):
    preds = torch.tensor([[[0.5]], [[1.0]], [[2.0]]], dtype=torch.float64)
    log_prior = torch.tensor([-2.0, -3.0, -1.0], dtype=torch.float64)
    log_posterior = torch.tensor([-1.0, -1.0, -2.0], dtype=torch.float64)
    kl_weight = 0.7
    criterion = VariationalLoss(
        SquaredErrorLikelihood(),
        alpha=alpha,
        num_batches=num_batches,
        kl_weight=kl_weight,
    )

    loss = criterion(
        preds,
        torch.zeros(1, 1, dtype=torch.float64),
        log_prior=log_prior,
        log_posterior=log_posterior,
    )

    log_joints = [
        -num_batches * value**2 + kl_weight * ratio
        for value, ratio in zip([0.5, 1.0, 2.0], [-1.0, -2.0, 1.0])
    ]
    if alpha == 1:
        expected = -sum(log_joints) / (len(log_joints) * num_batches)
    else:
        importance_mean = sum(
            math.exp((1 - alpha) * value) for value in log_joints
        ) / len(log_joints)
        expected = -math.log(importance_mean) / ((1 - alpha) * num_batches)
    assert loss.item() == pytest.approx(expected, abs=1e-12)


def test_renyi_keeps_weight_ratios_paired_with_predictions():
    preds = torch.tensor([[[0.0]], [[2.0]]], dtype=torch.float64)
    target = torch.zeros(1, 1, dtype=torch.float64)
    log_prior = torch.tensor([-4.0, 0.0], dtype=torch.float64)
    log_posterior = torch.zeros(2, dtype=torch.float64)
    criterion = VariationalLoss(SquaredErrorLikelihood(), alpha=0.5)

    paired = criterion(preds, target, log_prior=log_prior, log_posterior=log_posterior)
    shuffled = criterion(
        preds, target, log_prior=log_prior.flip(0), log_posterior=log_posterior
    )

    assert paired.item() == pytest.approx(4.0)
    assert shuffled.item() == pytest.approx(-2 * math.log((1 + math.exp(-4)) / 2))
    assert not torch.isclose(paired, shuffled)


def test_renyi_gradients_use_sample_importance_weights():
    preds = torch.tensor([[[0.5]], [[1.0]], [[2.0]]], dtype=torch.float64)
    preds.requires_grad_()
    log_prior = torch.tensor(
        [-2.0, -3.0, -1.0], dtype=torch.float64, requires_grad=True
    )
    log_posterior = torch.tensor(
        [-1.0, -1.0, -2.0], dtype=torch.float64, requires_grad=True
    )
    criterion = VariationalLoss(
        SquaredErrorLikelihood(), alpha=0.5, num_batches=3, kl_weight=0.7
    )

    loss = criterion(
        preds,
        torch.zeros(1, 1, dtype=torch.float64),
        log_prior=log_prior,
        log_posterior=log_posterior,
    )
    loss.backward()

    log_joints = torch.tensor([-1.45, -4.4, -11.3], dtype=torch.float64)
    importance = torch.softmax(0.5 * log_joints, dim=0)
    torch.testing.assert_close(
        preds.grad.flatten(), 2 * preds.detach().flatten() * importance
    )
    torch.testing.assert_close(log_prior.grad, -0.7 / 3 * importance)
    torch.testing.assert_close(log_posterior.grad, 0.7 / 3 * importance)


@pytest.mark.parametrize("alpha", [1 - 1e-7, 1 + 1e-7, 1 - 1e-9, 1 + 1e-9])
def test_renyi_near_one_is_stable_in_float32(alpha):
    log_prior = torch.tensor([-10000.0, 0.0, 10000.0], requires_grad=True)
    loss = VariationalLoss(ZeroLikelihood(), alpha=alpha)(
        torch.zeros(3, 1, 1),
        torch.zeros(1, 1),
        log_prior=log_prior,
        log_posterior=torch.zeros(3),
    )
    loss.backward()

    scaled = (1 - alpha) * log_prior.detach().double()
    reference = -torch.log1p(torch.expm1(scaled).mean()) / (1 - alpha)
    assert loss.item() == pytest.approx(reference.item(), rel=2e-5, abs=2e-4)
    torch.testing.assert_close(
        log_prior.grad, -torch.softmax(scaled, dim=0).float(), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("classification", [False, True])
def test_renyi_sums_likelihood_over_batch_and_outputs(classification):
    preds = torch.tensor(
        [[[0.2, -0.4], [1.0, 0.5]], [[-0.3, 0.8], [0.7, -0.2]]],
        dtype=torch.float64,
    )
    if classification:
        likelihood = nn.CrossEntropyLoss(reduction="none")
        target = torch.tensor([0, 1])
    else:
        likelihood = SquaredErrorLikelihood()
        target = torch.tensor([[0.5, -0.5], [1.5, 0.2]], dtype=torch.float64)
    log_prior = torch.tensor([-2.0, -4.0], dtype=torch.float64)
    log_posterior = torch.tensor([-1.0, -2.0], dtype=torch.float64)

    loss = VariationalLoss(likelihood, alpha=0.5)(
        preds, target, log_prior=log_prior, log_posterior=log_posterior
    )

    log_joints = [
        -likelihood(preds[index], target).sum().item()
        + log_prior[index].item()
        - log_posterior[index].item()
        for index in range(len(preds))
    ]
    expected = -2 * math.log(sum(math.exp(value / 2) for value in log_joints) / 2)
    assert loss.item() == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_regression_sum_reduction_matches_elementwise_likelihood(alpha):
    preds = torch.tensor(
        [
            [[0.2, -0.4], [1.0, 0.5]],
            [[-0.3, 0.8], [0.7, -0.2]],
            [[0.9, 0.4], [-0.1, 0.3]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor([[0.5, -0.5], [1.5, 0.2]], dtype=torch.float64)
    log_prior = torch.tensor([-2.0, -4.0, -3.0], dtype=torch.float64)
    log_posterior = torch.tensor([-1.0, -2.0, -1.5], dtype=torch.float64)
    losses = []
    gradients = []
    for reduction in ("none", "sum"):
        loss = VariationalLoss(nn.MSELoss(reduction=reduction), alpha=alpha)(
            preds, target, log_prior=log_prior, log_posterior=log_posterior
        )
        losses.append(loss)
        gradients.append(torch.autograd.grad(loss, preds)[0])

    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(gradients[0], gradients[1])


def test_renyi_rejects_analytic_kl_without_sample_log_probabilities():
    criterion = VariationalLoss(ZeroLikelihood(), alpha=0.5)
    with pytest.raises(ValueError):
        criterion(torch.zeros(3, 1, 1), torch.zeros(1, 1), torch.tensor(0.5))


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"log_prior": torch.zeros(3)},
        {"log_posterior": torch.zeros(3)},
        {"log_prior": torch.tensor(0.0), "log_posterior": torch.zeros(3)},
        {"log_prior": torch.zeros(3), "log_posterior": torch.zeros(2)},
        {"log_prior": torch.zeros(3, 1), "log_posterior": torch.zeros(3)},
        {
            "kl_divergence": torch.tensor(0.5),
            "log_prior": torch.zeros(3),
            "log_posterior": torch.zeros(3),
        },
    ],
)
def test_renyi_rejects_missing_misaligned_or_ambiguous_log_probabilities(arguments):
    criterion = VariationalLoss(ZeroLikelihood(), alpha=0.5)
    with pytest.raises(ValueError):
        criterion(torch.zeros(3, 1, 1), torch.zeros(1, 1), **arguments)
