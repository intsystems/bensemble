import abc

import torch.nn as nn


class SearchSpace(abc.ABC):
    @abc.abstractmethod
    def sample(self) -> dict:
        """Sample a random architecture config uniformly from the space."""
        ...

    @abc.abstractmethod
    def mutate(self, config: dict) -> dict:
        """Return a new config that is a mutated copy of the given config.
        Must not modify the input config in place."""
        ...

    @abc.abstractmethod
    def build(self, config: dict) -> nn.Module:
        """Instantiate and return an untrained nn.Module for the given config."""
        ...
