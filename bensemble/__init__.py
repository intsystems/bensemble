"""
bensemble: Bayesian ensemble methods for neural networks.

Modules:
- core: базовые классы и утилиты
- methods: реализация методов вариационного и байесовского ансамбля
"""

__version__ = "0.1.0"

from .core import *
from .methods import *

__all__ = [
    "core",
    "methods",
]
