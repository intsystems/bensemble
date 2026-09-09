import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bensemble.search.selection import (
    classification_nll_criterion,
    forward_select,
    regression_mse_criterion,
)


class _FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits.unsqueeze(0).expand(x.shape[0], -1)


class _FixedRegressionModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self.value)


def _make_clf_loader(n=8, num_classes=2, batch_size=4):
    x = torch.randn(n, 4)
    y = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _make_reg_loader(n=8, batch_size=4):
    x = torch.randn(n, 4)
    y = torch.ones(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


_DEVICE = torch.device("cpu")


def test_classification_nll_criterion_return_type():
    """Criterion returns a positive float."""
    members = [_FixedLogitModel(torch.tensor([1.0, 0.5]))] * 2
    result = classification_nll_criterion(members, _make_clf_loader(), _DEVICE)
    assert isinstance(result, float)
    assert result > 0


def test_classification_nll_criterion_near_perfect():
    """Near-perfect classifier yields near-zero NLL."""
    # class 0 strongly predicted, all targets are class 0
    members = [_FixedLogitModel(torch.tensor([10.0, -10.0]))]
    result = classification_nll_criterion(members, _make_clf_loader(), _DEVICE)
    assert result < 0.1


def test_classification_nll_criterion_multiple_batches():
    """Criterion handles multi-batch loaders correctly."""
    loader = _make_clf_loader(n=16, batch_size=4)
    members = [_FixedLogitModel(torch.tensor([1.0, 0.0]))] * 2
    result = classification_nll_criterion(members, loader, _DEVICE)
    assert isinstance(result, float)
    assert result > 0


def test_regression_mse_criterion_zero_error():
    """Perfect predictions yield zero MSE."""
    members = [_FixedRegressionModel(1.0)]
    result = regression_mse_criterion(members, _make_reg_loader(), _DEVICE)
    assert abs(result) < 1e-5


def test_regression_mse_criterion_unit_error():
    """Predictions off by 1 yield unit MSE."""
    members = [_FixedRegressionModel(0.0)]
    result = regression_mse_criterion(members, _make_reg_loader(), _DEVICE)
    assert abs(result - 1.0) < 1e-5


def test_forward_select_correct_count():
    """Forward selection returns exactly the requested number of members."""
    pool = [_FixedLogitModel(torch.tensor([float(i), 0.0])) for i in range(4)]
    result = forward_select(
        pool, _make_clf_loader(), 2, _DEVICE, classification_nll_criterion
    )
    assert len(result) == 2


def test_forward_select_no_repeats():
    """Forward selection never picks the same model twice."""
    pool = [_FixedLogitModel(torch.tensor([float(i), 0.0])) for i in range(4)]
    result = forward_select(
        pool, _make_clf_loader(), 3, _DEVICE, classification_nll_criterion
    )
    ids = [id(m) for m in result]
    assert len(set(ids)) == 3


def test_forward_select_picks_best_member():
    """Greedy selection picks the member with the lowest NLL."""
    good = _FixedLogitModel(torch.tensor([10.0, -10.0]))
    bad = _FixedLogitModel(torch.tensor([-10.0, 10.0]))
    pool = [good, bad]
    result = forward_select(
        pool, _make_clf_loader(), 1, _DEVICE, classification_nll_criterion
    )
    assert result[0] is good


def _shuffled_loader(x, y, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y), batch_size=16, shuffle=True, generator=generator
    )


def test_classification_nll_criterion_is_order_invariant():
    """
    Two copies of one model must score exactly like that model alone, even when
    the validation loader reshuffles between passes over the members.
    """
    torch.manual_seed(0)
    x, y = torch.randn(64, 4), torch.randint(0, 2, (64,))
    model = nn.Linear(4, 2).eval()

    single = classification_nll_criterion([model], _shuffled_loader(x, y), _DEVICE)
    duplicated = classification_nll_criterion(
        [model, model], _shuffled_loader(x, y, seed=1), _DEVICE
    )
    ordered = classification_nll_criterion(
        [model], DataLoader(TensorDataset(x, y), batch_size=16), _DEVICE
    )

    assert duplicated == pytest.approx(single)
    assert single == pytest.approx(ordered)


def test_regression_mse_criterion_is_order_invariant():
    """
    Two copies of one regressor must score exactly like the regressor alone under
    a shuffling validation loader.
    """
    torch.manual_seed(0)
    x, y = torch.randn(64, 4), torch.randn(64, 1)
    model = nn.Linear(4, 1).eval()

    single = regression_mse_criterion([model], _shuffled_loader(x, y), _DEVICE)
    duplicated = regression_mse_criterion(
        [model, model], _shuffled_loader(x, y, seed=1), _DEVICE
    )

    assert duplicated == pytest.approx(single)
