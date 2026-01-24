import pytest
import torch


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def input_data(batch_size):
    return torch.randn(batch_size, 10)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
