import torch
import torch.nn.functional as F

from bensemble.calibration.scaling import TemperatureScaling


def test_initialization():
    """Default and custom initialization work properly."""
    scaler_default = TemperatureScaling()
    assert scaler_default.temperature.item() == 1.5

    scaler_custom = TemperatureScaling(init_temp=2.0)
    assert scaler_custom.temperature.item() == 2.0


def test_forward_pass():
    """Logits are divided by T."""
    scaler = TemperatureScaling(init_temp=2.0)

    logits = torch.tensor([[10.0, 4.0], [2.0, -2.0]])
    expected = torch.tensor([[5.0, 2.0], [1.0, -1.0]])

    scaled_logits = scaler(logits)
    assert torch.allclose(scaled_logits, expected)


def test_accuracy_preservation():
    """
    Temperature scaling doesn't change the predicted class.
    """
    scaler = TemperatureScaling(init_temp=5.0)

    logits = torch.randn(100, 10)
    original_preds = logits.argmax(dim=1)

    scaled_logits = scaler(logits)
    calibrated_preds = scaled_logits.argmax(dim=1)

    assert torch.equal(original_preds, calibrated_preds), (
        "Temperature Scaling changed the predictions."
    )


def test_fit_optimization():
    """
    'fit' method works properly, changes temperature and lowers Negative Log-Likelihood.
    """
    torch.manual_seed(42)

    logits = torch.randn(200, 5) * 10.0
    labels = torch.randint(0, 5, (200,))

    scaler = TemperatureScaling(init_temp=1.0)

    initial_loss = F.cross_entropy(logits, labels).item()

    scaler.fit(logits, labels, max_iter=20)

    assert scaler.temperature.item() != 1.0

    calibrated_logits = scaler(logits)
    final_loss = F.cross_entropy(calibrated_logits, labels).item()

    assert final_loss < initial_loss, (
        "Calibration failed to improve the Negative Log-Likelihood."
    )
