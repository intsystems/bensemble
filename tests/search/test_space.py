import torch.nn as nn

from bensemble.search.space import SearchSpace


class _MinimalSpace(SearchSpace):
    def sample(self) -> dict:
        return {"units": 4}

    def mutate(self, config: dict) -> dict:
        return {"units": config["units"] + 1}

    def build(self, config: dict) -> nn.Module:
        return nn.Linear(config["units"], 1)


def test_subclass_instantiation():
    """SearchSpace subclass can be instantiated without errors."""
    space = _MinimalSpace()
    assert isinstance(space, SearchSpace)


def test_sample_returns_dict():
    """sample() returns a configuration dict."""
    space = _MinimalSpace()
    assert isinstance(space.sample(), dict)


def test_mutate_returns_new_dict():
    """mutate() returns a different dict object with changed values."""
    space = _MinimalSpace()
    config = space.sample()
    mutated = space.mutate(config)
    assert mutated is not config
    assert mutated != config


def test_build_returns_module():
    """build() returns an nn.Module from the config."""
    space = _MinimalSpace()
    model = space.build(space.sample())
    assert isinstance(model, nn.Module)
