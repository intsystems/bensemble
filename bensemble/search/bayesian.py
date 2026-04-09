from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bensemble.core.ensemble import Ensemble
from bensemble.search.selection import classification_nll_criterion
from bensemble.search.space import SearchSpace


@dataclass
class _Candidate:
    model: nn.Module
    score: float
    probs: torch.Tensor

class NESBayesianSampler:
    
    """Neural Ensemble Search via Bayesian Sampling (NESBS, Shu et al., UAI 2022).
    This implementation follows the paper's practical recipe:
    1) build a candidate model pool from a search space;
    2) estimate a posterior over candidates from validation losses;
    3) select a diverse final ensemble either by:
       - weighted Monte Carlo sampling, or
       - an SVGD-inspired iterative refinement with diversity regularization.
    """

    def __init__(
        self,
        space: SearchSpace,
        train_fn: Callable[[nn.Module], None],
        pool_size: int = 50,
        ensemble_size: int = 5,
        temperature: float = 1.0,
        diversity_weight: float = 0.5,
        svgd_steps: int = 20,
        svgd_lr: float = 0.1,
        device: Optional[torch.device] = None,
        criterion: Optional[
            Callable[[list[nn.Module], DataLoader, torch.device], float]
        ] = None,
    ) -> None:
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1.")
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1.")
        if ensemble_size > pool_size:
            raise ValueError("ensemble_size must be <= pool_size.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if svgd_steps < 1:
            raise ValueError("svgd_steps must be >= 1.")

        self.space = space
        self.train_fn = train_fn
        self.pool_size = pool_size
        self.ensemble_size = ensemble_size
        self.temperature = temperature
        self.diversity_weight = diversity_weight
        self.svgd_steps = svgd_steps
        self.svgd_lr = svgd_lr
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.criterion = (
            criterion if criterion is not None else classification_nll_criterion
        )

    def _predict_probs(self, model: nn.Module, val_loader: DataLoader) -> torch.Tensor:
        model.to(self.device)
        model.eval()
        probs_batches: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                logits = model(x)
                probs_batches.append(F.softmax(logits, dim=-1).cpu())
        return torch.cat(probs_batches, dim=0)
    
    def _build_pool(self, val_loader: DataLoader) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for _ in range(self.pool_size):
            config = self.space.sample()
            model = self.space.build(config)
            self.train_fn(model)
            score = self.criterion([model], val_loader, self.device)
            probs = self._predict_probs(model, val_loader)
            candidates.append(_Candidate(model=model, score=score, probs=probs))
        return candidates

    def _posterior_probs(self, candidates: list[_Candidate]) -> torch.Tensor:
        scores = torch.tensor([c.score for c in candidates], dtype=torch.float32)
        centered = scores - scores.min()
        logits = -centered / self.temperature
        return torch.softmax(logits, dim=0)

    @staticmethod
    def _pairwise_diversity(a_probs: torch.Tensor, b_probs: torch.Tensor) -> float:
        eps = 1e-8
        p = a_probs.clamp_min(eps)
        q = b_probs.clamp_min(eps)
        kl_pq = (p * (p.log() - q.log())).sum(dim=-1).mean()
        kl_qp = (q * (q.log() - p.log())).sum(dim=-1).mean()
        return float(0.5 * (kl_pq + kl_qp))

    def sample_mc(self, val_loader: DataLoader) -> Ensemble:
        """
        Args:
            val_loader (DataLoader): Used to evaluate the posterior.

        Returns:
            Ensemble: The final ensemble wrapped in bensemble's core abstraction.
        """

        candidates = self._build_pool(val_loader)
        probs = self._posterior_probs(candidates)
        chosen = torch.multinomial(
            probs, num_samples=self.ensemble_size, replacement=False
        )
        models = [candidates[idx].model for idx in chosen.tolist()]
        return Ensemble.from_models(models)


    def sample_svgd(self, val_loader: DataLoader) -> Ensemble:
        """
        Args:
            val_loader (DataLoader): Used to evaluate the architecture's loss/posterior.

        Returns:
            Ensemble: The final ensemble wrapped in bensemble's core abstraction.
        """
        candidates = self._build_pool(val_loader)
        posterior = self._posterior_probs(candidates)
        n = len(candidates)
        logits = torch.log(posterior + 1e-8)

        particles = torch.multinomial(
            posterior, num_samples=self.ensemble_size, replacement=False
        )
        for _ in range(self.svgd_steps):
            for i in range(self.ensemble_size):
                current = particles[i].item()
                best_idx = current
                best_value = float("-inf")

                for candidate_idx in range(n):
                    if candidate_idx in particles.tolist() and candidate_idx != current:
                        continue
                    repulsion = 0.0
                    for j in range(self.ensemble_size):
                        if j == i:
                            continue
                        other_idx = particles[j].item()
                        div = self._pairwise_diversity(
                            candidates[candidate_idx].probs,
                            candidates[other_idx].probs,
                        )
                        repulsion += div
                    value = logits[candidate_idx].item() + (
                        self.svgd_lr * self.diversity_weight * repulsion
                    )
                    if value > best_value:
                        best_value = value
                        best_idx = candidate_idx
                particles[i] = best_idx

        models = [candidates[idx].model for idx in particles.tolist()]
        return Ensemble.from_models(models)