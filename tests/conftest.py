import pytest
import torch


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def num_classes():
    return 3


@pytest.fixture
def ensemble_size():
    return 5


@pytest.fixture
def input_data(batch_size):
    return torch.randn(batch_size, 10)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def ensemble_probs(ensemble_size, batch_size, num_classes):
    """
    Generates probabilities from a random ensemble.
    Shape: [Ensemble, Batch, Classes].
    """
    return torch.softmax(torch.randn(ensemble_size, batch_size, num_classes), dim=-1)
