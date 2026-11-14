import pytest
import torch
from torch import nn

from bensemble.core.ensemble import Ensemble
from bensemble.layers import BayesianLinear
from bensemble.utils import predict_with_uncertainty


def _model():
    return nn.Sequential(
        nn.Sequential(BayesianLinear(4, 4), nn.BatchNorm1d(4)), nn.Dropout(0.5)
    )


def _modes(model):
    return {name: layer.training for name, layer in model.named_modules()}


def _predict(kind, model, inputs):
    if kind == "explicit":
        return Ensemble.from_models([model]).predict_members(inputs)
    if kind == "stochastic":
        return Ensemble.from_stochastic(
            model, num_samples=6, mode="both"
        ).predict_members(inputs)
    return predict_with_uncertainty(model, inputs, num_samples=6)


@pytest.mark.parametrize("kind", ["explicit", "stochastic", "utility"])
@pytest.mark.parametrize("root_training", [False, True])
def test_prediction_preserves_mixed_modes_and_frozen_batchnorm(kind, root_training):
    torch.manual_seed(0)
    model = _model()
    model.train(root_training)
    model[0][1].eval()
    if not root_training:
        model[0][0].train()
        model[1].train()
    before = _modes(model)
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    observations = []

    def record_modes(module, inputs):
        observations.append(_modes(module))

    handle = model.register_forward_pre_hook(record_modes)
    inputs = torch.randn(8, 4)
    try:
        predictions = _predict(kind, model, inputs)
    finally:
        handle.remove()

    assert _modes(model) == before
    for observation in observations:
        assert not observation[""]
        assert not observation["0"]
        assert not observation["0.1"]
        assert observation["0.0"] == (kind != "explicit")
        assert observation["1"] == (kind == "stochastic")
    if kind == "stochastic":
        assert predictions.var(dim=0).sum() > 0
    elif kind == "utility":
        assert predictions[1].max() > 0

    # A subsequent training forward must still leave the frozen BN untouched.
    model(inputs)
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, buffers[name])


@pytest.mark.parametrize("kind", ["explicit", "stochastic", "utility"])
def test_prediction_restores_modes_after_forward_error(kind):
    class Fail(nn.Module):
        def forward(self, inputs):
            raise RuntimeError("forward failed")

    model = _model()
    model.append(Fail())
    model.train()
    model[0][1].eval()
    model[1].eval()
    before = _modes(model)
    buffers = {name: value.clone() for name, value in model.named_buffers()}

    with pytest.raises(RuntimeError, match="forward failed"):
        _predict(kind, model, torch.randn(8, 4))

    assert _modes(model) == before
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, buffers[name])


@pytest.mark.parametrize("fail", [False, True])
def test_explicit_members_restore_shared_layers_and_all_members(fail):
    class MaybeFail(nn.Module):
        def forward(self, inputs):
            if fail:
                raise RuntimeError("second member failed")
            return inputs

    batch_norm = nn.BatchNorm1d(4)
    first = nn.Sequential(batch_norm, nn.Dropout(0.5)).train()
    second = nn.Sequential(batch_norm, MaybeFail()).eval()
    batch_norm.train()
    before = [_modes(first), _modes(second)]
    ensemble = Ensemble.from_models([first, second])

    if fail:
        with pytest.raises(RuntimeError, match="second member failed"):
            ensemble.predict_members(torch.randn(8, 4))
    else:
        ensemble.predict_members(torch.randn(8, 4))

    assert [_modes(first), _modes(second)] == before
    assert batch_norm.num_batches_tracked.item() == 0


@pytest.mark.parametrize("kind", ["explicit", "stochastic", "utility"])
def test_prediction_restores_modes_when_eval_raises(kind):
    class FailingEval(nn.Sequential):
        fail = False

        def train(self, mode=True):
            super().train(mode)
            if self.fail and not mode:
                raise RuntimeError("eval failed")
            return self

    model = FailingEval(BayesianLinear(4, 4), nn.BatchNorm1d(4))
    model.train()
    model[1].eval()
    model.fail = True
    before = _modes(model)

    with pytest.raises(RuntimeError, match="eval failed"):
        _predict(kind, model, torch.randn(8, 4))

    assert _modes(model) == before
