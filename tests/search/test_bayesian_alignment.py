import pytest
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence
from torch.utils.data import DataLoader, TensorDataset

from bensemble.search.bayesian import NESBayesianSampler
from bensemble.search.space import SearchSpace


class _ScaledLogits(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * x


class _ScaleSpace(SearchSpace):
    def __init__(self, scales):
        self.scales = iter(scales)

    def sample(self):
        return {"scale": next(self.scales)}

    def mutate(self, config):
        return config.copy()

    def build(self, config):
        return _ScaledLogits(**config)


def _sampler(scales, **kwargs):
    return NESBayesianSampler(
        space=_ScaleSpace(scales),
        train_fn=kwargs.pop("train_fn", lambda model: None),
        pool_size=len(scales),
        ensemble_size=2,
        device=torch.device("cpu"),
        **kwargs,
    )


def _inputs():
    x = torch.linspace(-3.0, 3.0, 17, dtype=torch.float64)
    return torch.stack((x, -x), dim=-1)


def _loader(x, shuffle):
    return DataLoader(
        TensorDataset(x, x.argmax(dim=-1)),
        batch_size=6,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(0),
    )


@pytest.mark.parametrize("shuffle", [False, True])
def test_identical_candidates_have_zero_diversity(shuffle):
    sampler = _sampler([1.0, 1.0])
    a, b = sampler._build_pool(_loader(_inputs(), shuffle))

    torch.testing.assert_close(a.probs, b.probs, rtol=0, atol=0)
    assert sampler._pairwise_diversity(a.probs, b.probs) == 0.0


@pytest.mark.parametrize("shuffle", [False, True])
def test_diversity_matches_kl_on_the_same_examples(shuffle):
    x = _inputs()
    sampler = _sampler([1.0, 0.4])
    a, b = sampler._build_pool(_loader(x, shuffle))
    p, q = Categorical(logits=x), Categorical(logits=0.4 * x)
    expected = 0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).mean()

    assert sampler._pairwise_diversity(a.probs, b.probs) == pytest.approx(
        expected.item()
    )


class _ChangingInputs(TensorDataset):
    def __init__(self, x):
        super().__init__(x, x.argmax(dim=-1))
        self.visits = [0] * len(x)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        self.visits[index] += 1
        return x * self.visits[index], y


def test_candidates_share_validation_transforms():
    loader = DataLoader(_ChangingInputs(_inputs()), batch_size=6)
    sampler = _sampler([1.0, 1.0], criterion=lambda members, loader, device: 0.0)
    a, b = sampler._build_pool(loader)

    torch.testing.assert_close(a.probs, b.probs, rtol=0, atol=0)


def test_custom_criterion_receives_trained_candidates_and_preserves_scores():
    loader = _loader(_inputs(), shuffle=True)
    trained = []
    scored = []

    def criterion(members, val_loader, device):
        assert val_loader is loader
        assert device == torch.device("cpu")
        assert len(members) == 1
        model = members[0]
        assert model in trained
        scored.append(model)
        return model.scale + 2.0

    sampler = _sampler([1.0, 0.4], train_fn=trained.append, criterion=criterion)
    candidates = sampler._build_pool(loader)

    assert [candidate.model for candidate in candidates] == trained == scored
    assert [candidate.score for candidate in candidates] == [3.0, 2.4]
