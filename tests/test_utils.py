import torch
import torch.nn as nn

from bensemble.utils import compute_uncertainty, enable_dropout, EarlyStopping


def test_uncertainty_regression():
    """
    Tests uncertainty calculation for regression (2D tensor).
    Expects Epistemic > 0 and Aleatoric == 0.
    """

    preds = torch.zeros(10, 5)
    preds[::2] = 1.0
    preds[1::2] = 0.0

    epistemic, aleatoric = compute_uncertainty(preds)

    assert epistemic.shape == (5,)
    assert aleatoric.shape == (5,)

    assert torch.all(epistemic > 0)

    assert torch.all(aleatoric == 0)


def test_uncertainty_classification():
    """
    Tests uncertainty for classification (3D tensor).
    Expects both Epistemic and Aleatoric > 0.
    """
    logits = torch.randn(5, 2, 3)
    probs = torch.softmax(logits, dim=-1)

    epistemic, aleatoric = compute_uncertainty(probs)

    assert epistemic.shape == (2, 3)

    assert aleatoric.shape == (2,)

    assert torch.all(aleatoric >= 0)


def test_enable_dropout():
    """
    Checks if enable_dropout sets Dropout layers to train mode
    while keeping other layers in eval mode.
    """
    model = nn.Sequential(nn.Linear(10, 10), nn.Dropout(0.5), nn.Linear(10, 1))

    model.eval()
    assert not model[1].training
    assert not model[0].training

    enable_dropout(model)

    assert model[1].training is True, "Dropout layer must be in train mode"
    assert model[0].training is False, "Linear layer should remain in eval mode"


def test_early_stopping_improvement():
    """Tests that counter resets when loss improves."""
    es = EarlyStopping(patience=3, delta=0.0)

    es(10.0)
    assert es.counter == 0
    assert es.best_score == 10.0

    es(9.0)
    assert es.counter == 0
    assert es.best_score == 9.0
    assert not es.early_stop


def test_early_stopping_patience():
    """Tests that early_stop triggers after patience is exceeded."""
    es = EarlyStopping(patience=2, delta=0.0)

    es(10.0)

    es(10.1)
    assert es.counter == 1
    assert not es.early_stop

    es(10.2)
    assert es.counter == 2
    assert es.early_stop is True


def test_early_stopping_delta():
    """Tests that improvement must differ by at least delta."""
    es = EarlyStopping(patience=2, delta=1.0)

    es(10.0)

    es(9.5)
    assert es.counter == 1

    es(8.0)
    assert es.counter == 0
    assert es.best_score == 8.0
