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

    assert torch.isclose(aleatoric, torch.tensor([0.0]), atol=1e-5).all()

    expected_total = -(0.5 * math.log(0.5) + 0.5 * math.log(0.5))
    assert torch.isclose(total, torch.tensor([expected_total]), atol=1e-5).all()

    assert torch.isclose(epistemic, total, atol=1e-5).all()


def test_regression_decomposition():
    """Regression uncertainty decomposition follows the law of total variance."""
    means = torch.tensor([[[10.0]], [[20.0]]])
    variances = torch.tensor([[[1.0]], [[1.0]]])

    total, aleatoric, epistemic = decompose_regression_uncertainty(means, variances)

    assert torch.isclose(aleatoric, torch.tensor([1.0]))
    assert torch.isclose(epistemic, torch.tensor([25.0]))
    assert torch.isclose(total, torch.tensor([26.0]))


def test_classification_perfect_agreement_epistemic_near_zero():
    """When all members predict identically, epistemic uncertainty is zero."""
    single = torch.softmax(torch.tensor([2.0, 0.5, 0.5]), dim=0)
    # shape (M=5, batch=4, C=3)
    probs = single.unsqueeze(0).unsqueeze(0).expand(5, 4, 3)
    _, _, epistemic = decompose_classification_uncertainty(probs)
    assert (epistemic.abs() < 1e-5).all()


def test_classification_single_member():
    """With a single model, epistemic uncertainty is zero."""
    probs = torch.softmax(torch.randn(1, 8, 3), dim=-1)
    total, aleatoric, epistemic = decompose_classification_uncertainty(probs)
    assert (epistemic.abs() < 1e-5).all()
    assert torch.allclose(total, aleatoric, atol=1e-5)


def test_classification_nonnegative_components(ensemble_probs):
    """All uncertainty components are non-negative."""
    total, aleatoric, epistemic = decompose_classification_uncertainty(ensemble_probs)
    assert (total >= 0).all()
    assert (aleatoric >= 0).all()
    assert (epistemic >= -1e-5).all()


def test_regression_single_member():
    """With one model, epistemic uncertainty is zero."""
    means = torch.randn(1, 4, 1)
    variances = torch.rand(1, 4, 1)
    total, aleatoric, epistemic = decompose_regression_uncertainty(means, variances)
    assert (epistemic.abs() < 1e-6).all()
    assert torch.allclose(total, aleatoric, atol=1e-6)


def test_regression_shapes_multidim():
    """Handles multi-dimensional outputs."""
    means = torch.randn(3, 8, 4)
    variances = torch.rand(3, 8, 4)
    total, aleatoric, epistemic = decompose_regression_uncertainty(means, variances)
    for t in (total, aleatoric, epistemic):
        assert t.shape == (8, 4)
