"""
bensemble: Bayesian ensemble methods for neural networks.

Modules:
- core: core classes and utilities
- methods: implementation of variational and Bayesian ensemble methods
"""

__version__ = "0.2.1"

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
