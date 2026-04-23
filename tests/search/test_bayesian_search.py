import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bensemble.core.ensemble import Ensemble
from bensemble.search.bayesian import NESBayesianSampler
from bensemble.search.space import SearchSpace


class _ConstantLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits.unsqueeze(0).repeat(x.shape[0], 1)


class _ToySearchSpace(SearchSpace):
    def __init__(self) -> None:
        self._idx = 0
        self._configs = [
            {"logits": torch.tensor([2.5, 0.2])},
            {"logits": torch.tensor([2.2, 0.3])},
            {"logits": torch.tensor([0.2, 2.4])},
            {"logits": torch.tensor([1.2, 1.1])},
        ]

    def sample(self) -> dict:
        cfg = self._configs[self._idx % len(self._configs)]
        self._idx += 1
        return {"logits": cfg["logits"].clone()}

    def mutate(self, config: dict) -> dict:
        return config

    def build(self, config: dict) -> nn.Module:
        return _ConstantLogitModel(config["logits"])


def _train_noop(_: nn.Module) -> None:
    return None


def _make_val_loader() -> DataLoader:
    x = torch.randn(12, 3)
    y = torch.zeros(12, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_nes_bayesian_sampler_mc_returns_ensemble_of_expected_size():
    """MC sampling returns an Ensemble with the requested number of members."""
    torch.manual_seed(0)
    sampler = NESBayesianSampler(
        space=_ToySearchSpace(),
        train_fn=_train_noop,
        pool_size=4,
        ensemble_size=2,
        temperature=0.7,
    )
    ensemble = sampler.sample_mc(_make_val_loader())
    assert isinstance(ensemble, Ensemble)
    assert ensemble.num_members == 2


def test_nes_bayesian_sampler_svgd_returns_ensemble_of_expected_size():
    """SVGD sampling returns an Ensemble with the requested number of members."""
    torch.manual_seed(0)
    sampler = NESBayesianSampler(
        space=_ToySearchSpace(),
        train_fn=_train_noop,
        pool_size=4,
        ensemble_size=3,
        temperature=1.0,
        diversity_weight=0.8,
        svgd_steps=5,
        svgd_lr=0.2,
    )
    ensemble = sampler.sample_svgd(_make_val_loader())
    assert isinstance(ensemble, Ensemble)
    assert ensemble.num_members == 3


def test_init_invalid_pool_size():
    """pool_size <= 0 raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="pool_size"):
        NESBayesianSampler(space=_ToySearchSpace(), train_fn=_train_noop, pool_size=0)


def test_init_invalid_ensemble_size():
    """ensemble_size <= 0 raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="ensemble_size"):
        NESBayesianSampler(space=_ToySearchSpace(), train_fn=_train_noop, ensemble_size=0)


def test_init_ensemble_gt_pool():
    """ensemble_size > pool_size raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="ensemble_size"):
        NESBayesianSampler(
            space=_ToySearchSpace(), train_fn=_train_noop, pool_size=3, ensemble_size=10
        )


def test_init_invalid_temperature():
    """temperature <= 0 raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="temperature"):
        NESBayesianSampler(space=_ToySearchSpace(), train_fn=_train_noop, temperature=0)


def test_init_invalid_svgd_steps():
    """svgd_steps <= 0 raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="svgd_steps"):
        NESBayesianSampler(space=_ToySearchSpace(), train_fn=_train_noop, svgd_steps=0)


def test_posterior_probs_sum_to_one():
    """Posterior probabilities sum to 1 and are all non-negative."""
    from bensemble.search.bayesian import _Candidate

    sampler = NESBayesianSampler(
        space=_ToySearchSpace(),
        train_fn=_train_noop,
        pool_size=4,
        ensemble_size=2,
        temperature=1.0,
    )
    dummy_probs = torch.softmax(torch.randn(8, 2), dim=-1)
    candidates = [_Candidate(model=nn.Linear(1, 1), score=float(i), probs=dummy_probs) for i in range(4)]
    probs = sampler._posterior_probs(candidates)
    assert abs(probs.sum().item() - 1.0) < 1e-5
    assert (probs >= 0).all()