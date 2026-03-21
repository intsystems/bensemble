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
    # [2 models, 1 sample, 2 classes]
    probs = torch.tensor(
        [
            [[1.0, 0.0]],  # Model 1 is confident in class 0
            [[0.0, 1.0]],  # Model 2 is confident in class 1
        ]
    )

    total, aleatoric, epistemic = decompose_classification_uncertainty(probs)

    # Aleatoric uncertainty is zero since each model is perfectly confident
    assert torch.isclose(aleatoric, torch.tensor([0.0]), atol=1e-5)

    # Total uncertainty equals entropy of the mean prediction [0.5, 0.5]
    expected_total = -(0.5 * math.log(0.5) + 0.5 * math.log(0.5))
    assert torch.isclose(total, torch.tensor([expected_total]), atol=1e-5)

    # Epistemic uncertainty equals total uncertainty in case of perfect disagreement
    assert torch.isclose(epistemic, total, atol=1e-5)


def test_regression_decomposition():
    """Regression uncertainty decomposition follows the law of total variance."""
    # [2 models, 1 sample, 1 output]
    means = torch.tensor([[[10.0]], [[20.0]]])
    variances = torch.tensor([[[1.0]], [[1.0]]])  # Identical observation noise

    total, aleatoric, epistemic = decompose_regression_uncertainty(means, variances)

    # Aleatoric uncertainty equals the mean predicted variance
    assert torch.isclose(aleatoric, torch.tensor([1.0]))

    # Epistemic uncertainty equals variance of ensemble means
    # Var([10, 20]) = ((10 - 15)^2 + (20 - 15)^2) / 2 = 25.0
    # (biased variance is used: unbiased=False)
    assert torch.isclose(epistemic, torch.tensor([25.0]))

    # Total uncertainty equals aleatoric + epistemic
    assert torch.isclose(total, torch.tensor([26.0]))
