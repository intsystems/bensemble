import torch.nn as nn

from bensemble.utils import enable_dropout


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
