"""
bensemble: Bayesian ensemble methods for neural networks.

Modules:
- core: базовые классы и утилиты
- methods: реализация методов вариационного и байесовского ансамбля
"""

__version__ = "0.1.0"


from .methods import (
    LaplaceApproximation,
    ProbabilisticBackpropagation,
)

__all__ = [
    "LaplaceApproximation",
    "ProbabilisticBackpropagation",
]
