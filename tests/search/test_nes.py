import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bensemble.core.ensemble import Ensemble
from bensemble.search.nes import EvolutionarySearcher, RandomSearcher
from bensemble.search.selection import classification_nll_criterion
from bensemble.search.space import SearchSpace


class _ToySpace(SearchSpace):
    _logits = [
        torch.tensor([2.5, 0.2]),
        torch.tensor([2.2, 0.3]),
        torch.tensor([0.2, 2.4]),
        torch.tensor([1.2, 1.1]),
        torch.tensor([1.8, 0.5]),
        torch.tensor([0.5, 1.9]),
    ]

    def __init__(self):
        self._idx = 0

    def sample(self) -> dict:
        cfg = {"logits": self._logits[self._idx % len(self._logits)].clone()}
        self._idx += 1
        return cfg

    def mutate(self, config: dict) -> dict:
        return {"logits": config["logits"] + 0.01}

    def build(self, config: dict) -> nn.Module:
        class _M(nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = nn.Parameter(logits.clone(), requires_grad=False)

            def forward(self, x):
                return self.logits.unsqueeze(0).expand(x.shape[0], -1)

        return _M(config["logits"])


def _noop_train(_: nn.Module) -> None:
    return None


def _make_val_loader() -> DataLoader:
    x = torch.randn(8, 3)
    y = torch.zeros(8, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_random_searcher_defaults():
    """RandomSearcher has the expected default hyperparameters."""
    searcher = RandomSearcher(space=_ToySpace(), train_fn=_noop_train)
    assert searcher.pool_size == 50
    assert searcher.ensemble_size == 5
    assert searcher.criterion is classification_nll_criterion


def test_random_searcher_custom_criterion():
    """Custom criterion is stored on the searcher."""

    def custom(m, v, d):
        return 0.0

    searcher = RandomSearcher(space=_ToySpace(), train_fn=_noop_train, criterion=custom)
    assert searcher.criterion is custom


def test_random_searcher_returns_ensemble():
    """search returns an Ensemble of the correct size with best-scoring members."""
    searcher = RandomSearcher(
        space=_ToySpace(),
        train_fn=_noop_train,
        pool_size=4,
        ensemble_size=2,
        device=torch.device("cpu"),
    )
    result = searcher.search(_make_val_loader())
    assert isinstance(result, Ensemble)
    assert result.num_members == 2
    x = torch.randn(4, 3)
    for member in result.member_modules:
        assert (member(x).argmax(dim=-1) == 0).all()


def test_random_searcher_shift_loader():
    """search accepts an optional shift validation loader."""
    searcher = RandomSearcher(
        space=_ToySpace(),
        train_fn=_noop_train,
        pool_size=4,
        ensemble_size=2,
        device=torch.device("cpu"),
    )
    result = searcher.search(_make_val_loader(), val_loader_shift=_make_val_loader())
    assert isinstance(result, Ensemble)


def test_evolutionary_searcher_defaults():
    """EvolutionarySearcher has the expected default hyperparameters."""
    searcher = EvolutionarySearcher(space=_ToySpace(), train_fn=_noop_train)
    assert searcher.population_size == 10
    assert searcher.num_parent_candidates == 3


def test_evolutionary_searcher_returns_ensemble():
    """search returns an Ensemble of the correct size."""
    searcher = EvolutionarySearcher(
        space=_ToySpace(),
        train_fn=_noop_train,
        pool_size=6,
        ensemble_size=2,
        population_size=3,
        num_parent_candidates=2,
        device=torch.device("cpu"),
    )
    result = searcher.search(_make_val_loader())
    assert isinstance(result, Ensemble)
    assert result.num_members == 2


def test_evolutionary_searcher_shift_loader():
    """search accepts an optional shift validation loader."""
    searcher = EvolutionarySearcher(
        space=_ToySpace(),
        train_fn=_noop_train,
        pool_size=6,
        ensemble_size=2,
        population_size=3,
        num_parent_candidates=2,
        device=torch.device("cpu"),
    )
    result = searcher.search(_make_val_loader(), val_loader_shift=_make_val_loader())
    assert isinstance(result, Ensemble)
