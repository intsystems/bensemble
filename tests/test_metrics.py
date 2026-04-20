import math

import pytest
import torch

from bensemble.metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_diagram,
)


@pytest.fixture
def random_probs(batch_size, num_classes):
    return torch.softmax(torch.randn(batch_size, num_classes), dim=-1)


@pytest.fixture
def random_targets(batch_size, num_classes):
    return torch.randint(0, num_classes, (batch_size,))


def test_nll_basic_shape(random_probs, random_targets):
    """NLL returns a positive float."""
    result = negative_log_likelihood(random_probs, random_targets)
    assert isinstance(result, float)
    assert result > 0


def test_nll_perfect_predictions():
    """NLL is near zero for perfect predictions."""
    probs = torch.tensor([[1.0, 0.0, 0.0]] * 4)
    targets = torch.zeros(4, dtype=torch.long)
    result = negative_log_likelihood(probs, targets)
    assert result < 1e-6


def test_nll_uniform_probs():
    """NLL for uniform probs equals log(num_classes)."""
    n, c = 8, 4
    probs = torch.ones(n, c) / c
    targets = torch.zeros(n, dtype=torch.long)
    result = negative_log_likelihood(probs, targets)
    assert abs(result - math.log(c)) < 1e-4


def test_brier_score_basic(random_probs, random_targets):
    """Brier score is a float in [0, 2]."""
    result = brier_score(random_probs, random_targets)
    assert isinstance(result, float)
    assert 0.0 <= result <= 2.0


def test_brier_score_perfect_zero():
    """Perfect predictions yield a Brier score of 0."""
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([0, 1, 0])
    assert brier_score(probs, targets) == 0.0


def test_brier_score_worst_case():
    """Worst-case predictions (certainty on wrong class) yield a Brier score of 2."""
    # predicts class 0 with certainty but truth is class 1
    probs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    targets = torch.tensor([1, 1])
    result = brier_score(probs, targets)
    assert abs(result - 2.0) < 1e-5


def test_ece_return_type_and_bounds(random_probs, random_targets):
    """ECE is a float in [0, 1]."""
    result = expected_calibration_error(random_probs, random_targets)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_ece_empty_bin_handling():
    """ECE is finite when most confidence bins are empty."""
    probs = torch.zeros(8, 2)
    probs[:, 0] = 0.99
    probs[:, 1] = 0.01
    targets = torch.zeros(8, dtype=torch.long)
    result = expected_calibration_error(probs, targets, n_bins=15)
    assert isinstance(result, float)
    assert math.isfinite(result)


def test_reliability_diagram_structure(random_probs, random_targets):
    """Output dict has the expected keys and the correct number of bins."""
    result = reliability_diagram(random_probs, random_targets, n_bins=5)
    assert set(result.keys()) == {"confidences", "accuracies", "proportions"}
    for key in result:
        assert len(result[key]) == 5


def test_reliability_diagram_proportions_non_negative(random_probs, random_targets):
    """All bin proportions are non-negative."""
    result = reliability_diagram(random_probs, random_targets, n_bins=5)
    assert all(p >= 0.0 for p in result["proportions"])


def test_reliability_diagram_custom_bins(random_probs, random_targets):
    """Custom n_bins controls the output list lengths."""
    result = reliability_diagram(random_probs, random_targets, n_bins=3)
    for key in result:
        assert len(result[key]) == 3
