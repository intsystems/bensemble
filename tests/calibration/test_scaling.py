import torch
import torch.nn.functional as F

from bensemble.calibration.scaling import TemperatureScaling
from bensemble.calibration.scaling import VectorScaling


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

    calibrated_logits = scaler(logits)
    final_loss = F.cross_entropy(calibrated_logits, labels).item()

    assert final_loss < initial_loss, (
        "Calibration failed to improve the Negative Log-Likelihood."
    )


def test_vector_scaling_initialization():
    """Vector Scaling initializes vectors correctly."""
    num_classes = 3
    scaler = VectorScaling(num_classes=num_classes)

    assert scaler.a.shape == (3,)
    assert scaler.b.shape == (3,)
    assert torch.allclose(scaler.a, torch.ones(3))
    assert torch.allclose(scaler.b, torch.zeros(3))


def test_vector_scaling_forward_pass():
    """The affine transformation (a * logits + b) works."""
    scaler = VectorScaling(num_classes=2)

    scaler.a.data = torch.tensor([2.0, 0.5])
    scaler.b.data = torch.tensor([1.0, -1.0])

    logits = torch.tensor([[10.0, 4.0], [2.0, -2.0]])

    expected = torch.tensor([[21.0, 1.0], [5.0, -2.0]])

    scaled_logits = scaler(logits)
    assert torch.allclose(scaled_logits, expected)


def test_vector_scaling_can_change_predictions():
    """Vector Scaling can change the argmax."""
    scaler = VectorScaling(num_classes=2)

    logits = torch.tensor([[1.0, 1.5]])
    original_preds = logits.argmax(dim=1)

    scaler.a.data = torch.tensor([1.0, 1.0])
    scaler.b.data = torch.tensor([5.0, 0.0])

    scaled_logits = scaler(logits)
    calibrated_preds = scaled_logits.argmax(dim=1)

    assert not torch.equal(original_preds, calibrated_preds), (
        "Vector Scaling should be able to change predictions, but it didn't."
    )


def test_vector_scaling_fit_optimization():
    """Vector Scaling reduces NLL during fit."""
    torch.manual_seed(42)
    num_classes = 5

    logits = torch.randn(200, num_classes) * 10.0
    labels = torch.randint(0, num_classes, (200,))

    scaler = VectorScaling(num_classes=num_classes)
    initial_loss = F.cross_entropy(logits, labels).item()

    scaler.fit(logits, labels, max_iter=20)

    calibrated_logits = scaler(logits)
    final_loss = F.cross_entropy(calibrated_logits, labels).item()

    assert final_loss < initial_loss, (
        "Vector Scaling failed to improve the Negative Log-Likelihood."
    )


def test_temperature_scaling_single_sample():
    """Single-sample input produces the correct output shape."""
    scaler = TemperatureScaling()
    logits = torch.randn(1, 5)
    out = scaler(logits)
    assert out.shape == (1, 5)


def test_temperature_scaling_large_temp_softens():
    """Large temperature softens the softmax confidence."""
    logits = torch.tensor([[10.0, 0.0]])
    scaler_sharp = TemperatureScaling(init_temp=1.0)
    scaler_flat = TemperatureScaling(init_temp=100.0)
    conf_sharp = torch.softmax(scaler_sharp(logits), dim=-1).max().item()
    conf_flat = torch.softmax(scaler_flat(logits), dim=-1).max().item()
    assert conf_flat < conf_sharp


def test_temperature_scaling_fit_returns_self():
    """fit() returns the scaler instance for method chaining."""
    scaler = TemperatureScaling()
    logits = torch.randn(50, 3)
    labels = torch.randint(0, 3, (50,))
    result = scaler.fit(logits, labels, max_iter=5)
    assert result is scaler


def test_vector_scaling_single_sample():
    """Single-sample input produces the correct output shape."""
    scaler = VectorScaling(num_classes=3)
    logits = torch.randn(1, 3)
    out = scaler(logits)
    assert out.shape == (1, 3)


def test_vector_scaling_fit_returns_self():
    """fit() returns the scaler instance for method chaining."""
    scaler = VectorScaling(num_classes=3)
    logits = torch.randn(50, 3)
    labels = torch.randint(0, 3, (50,))
    result = scaler.fit(logits, labels, max_iter=5)
    assert result is scaler
