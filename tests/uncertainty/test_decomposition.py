import torch
import math
from bensemble.uncertainty.decomposition import (
    decompose_classification_uncertainty,
    decompose_regression_uncertainty,
)


def test_uncertainty_decomposition_shapes(ensemble_probs, ensemble_size):
    """Outputs have shape (batch_size,) for all uncertainty components."""
    total, aleatoric, epistemic = decompose_classification_uncertainty(ensemble_probs)

    batch_size = ensemble_probs.shape[1]
    for tensor in (total, aleatoric, epistemic):
        assert tensor.shape == (batch_size,)

    assert ensemble_probs.shape[0] == ensemble_size


def test_perfect_disagreement():
    """
    Epistemic uncertainty is maximal when ensemble models confidently
    predict different classes for the same sample.
    """
    probs = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]
    )

    total, aleatoric, epistemic = decompose_classification_uncertainty(probs)

    assert torch.isclose(aleatoric, torch.tensor([0.0]), atol=1e-5)

    expected_total = -(0.5 * math.log(0.5) + 0.5 * math.log(0.5))
    assert torch.isclose(total, torch.tensor([expected_total]), atol=1e-5)

    assert torch.isclose(epistemic, total, atol=1e-5)


def test_regression_decomposition():
    """Regression uncertainty decomposition follows the law of total variance."""
    means = torch.tensor([[[10.0]], [[20.0]]])
    variances = torch.tensor([[[1.0]], [[1.0]]])

    total, aleatoric, epistemic = decompose_regression_uncertainty(means, variances)

    assert torch.isclose(aleatoric, torch.tensor([1.0]))
    assert torch.isclose(epistemic, torch.tensor([25.0]))
    assert torch.isclose(total, torch.tensor([26.0]))
