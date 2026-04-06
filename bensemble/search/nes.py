import random
from collections import deque
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bensemble.core.ensemble import Ensemble
from bensemble.search.selection import classification_nll_criterion, forward_select
from bensemble.search.space import SearchSpace


class RandomSearcher:
    """NES with Random Search (NES-RS, Algorithm 2 in Zaidi et al. NeurIPS 2021).

    Builds a pool of `pool_size` independently trained models with randomly
    sampled architectures, then applies greedy forward ensemble selection to
    pick the final ensemble of `ensemble_size` members.
    """

    def __init__(
        self,
        space: SearchSpace,
        pool_size: int = 50,
        ensemble_size: int = 5,
        train_fn: Optional[Callable[[nn.Module], None]] = None,
        device: Optional[torch.device] = None,
        criterion: Optional[
            Callable[[list[nn.Module], DataLoader, torch.device], float]
        ] = None,
    ) -> None:
        self.space = space
        self.pool_size = pool_size
        self.ensemble_size = ensemble_size
        self.train_fn = train_fn
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.criterion = (
            criterion if criterion is not None else classification_nll_criterion
        )

    def search(
        self,
        val_loader: DataLoader,
        val_loader_shift: Optional[DataLoader] = None,
    ) -> Ensemble:
        """Run NES-RS and return the selected ensemble.

        Args:
            val_loader: Validation loader used for ensemble selection.
            val_loader_shift: If provided, used instead of `val_loader` for
                the final ForwardSelect call (dataset-shift adaptation).

        Returns:
            Ensemble of `ensemble_size` members.
        """
        pool: list[nn.Module] = []
        for _ in range(self.pool_size):
            config = self.space.sample()
            model = self.space.build(config)
            self.train_fn(model)
            pool.append(model)

        selection_loader = (
            val_loader_shift if val_loader_shift is not None else val_loader
        )
        selected = forward_select(
            pool, selection_loader, self.ensemble_size, self.device, self.criterion
        )
        return Ensemble.from_models(selected)


class EvolutionarySearcher:
    """NES with Regularized Evolution (NES-RE, Algorithm 1 in Zaidi et al. NeurIPS 2021).

    Evolves a population of architectures using ensemble-aware parent selection
    (ForwardSelect on the population) and single-step mutation. The full history
    of trained models forms the pool from which the final ensemble is selected.
    """

    def __init__(
        self,
        space: SearchSpace,
        pool_size: int = 50,
        ensemble_size: int = 5,
        population_size: int = 10,
        num_parent_candidates: int = 3,
        train_fn: Optional[Callable[[nn.Module], None]] = None,
        device: Optional[torch.device] = None,
        criterion: Optional[
            Callable[[list[nn.Module], DataLoader, torch.device], float]
        ] = None,
    ) -> None:
        self.space = space
        self.pool_size = pool_size
        self.ensemble_size = ensemble_size
        self.population_size = population_size
        self.num_parent_candidates = num_parent_candidates
        self.train_fn = train_fn
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.criterion = (
            criterion if criterion is not None else classification_nll_criterion
        )

    def search(
        self,
        val_loader: DataLoader,
        val_loader_shift: Optional[DataLoader] = None,
    ) -> Ensemble:
        """Run NES-RE and return the selected ensemble.

        Args:
            val_loader: Validation loader used for parent selection and final
                ensemble selection.
            val_loader_shift: If provided, used instead of `val_loader` for
                the final ForwardSelect call (dataset-shift adaptation).

        Returns:
            Ensemble of `ensemble_size` members.
        """
        # Maps id(model) -> config to support mutation of selected parents.
        config_map: dict[int, dict] = {}

        def _build_and_train(config: dict) -> nn.Module:
            model = self.space.build(config)
            self.train_fn(model)
            config_map[id(model)] = config
            return model

        # --- Initialisation: seed population and pool ---
        population: deque[nn.Module] = deque()
        pool: list[nn.Module] = []

        for _ in range(self.population_size):
            model = _build_and_train(self.space.sample())
            population.append(model)
            pool.append(model)

        # --- Evolution loop ---
        while len(pool) < self.pool_size:
            # Select m parent candidates from the current population via ForwardSelect.
            parent_candidates = forward_select(
                list(population),
                val_loader,
                self.num_parent_candidates,
                self.device,
                self.criterion,
            )

            # Sample one parent uniformly at random.
            parent = random.choice(parent_candidates)
            parent_config = config_map[id(parent)]

            # Mutate and train child.
            child_config = self.space.mutate(parent_config)
            child = _build_and_train(child_config)

            population.append(child)
            pool.append(child)

            # Remove the oldest member from the population (regularized evolution).
            population.popleft()

        # --- Final ensemble selection ---
        selection_loader = (
            val_loader_shift if val_loader_shift is not None else val_loader
        )
        selected = forward_select(
            pool, selection_loader, self.ensemble_size, self.device, self.criterion
        )
        return Ensemble.from_models(selected)
