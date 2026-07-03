"""
bensemble: Bayesian ensemble methods for neural networks.

Modules:
- core: core classes and utilities
- methods: implementation of variational and Bayesian ensemble methods
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("bensemble")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .methods import (
    LaplaceApproximation,
    PBPEngine,
)
from .search import RandomSearcher, EvolutionarySearcher, SearchSpace

__all__ = [
    "LaplaceApproximation",
    "PBPEngine",
    "RandomSearcher",
    "EvolutionarySearcher",
    "SearchSpace",
]
