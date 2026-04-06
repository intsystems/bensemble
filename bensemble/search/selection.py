from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def classification_nll_criterion(
    members: list[nn.Module],
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Default criterion for classification tasks.
    Ensemble prediction = mean of per-member softmax probabilities.
    Score = mean NLL over val_loader.
    members: list of trained nn.Module in eval() mode.
    """
    with torch.no_grad():
        probs_list: list[torch.Tensor] = []
        labels: torch.Tensor | None = None

        for model in members:
            model.to(device)
            model.eval()
            member_probs: list[torch.Tensor] = []
            member_labels: list[torch.Tensor] = []

            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                member_probs.append(F.softmax(model(x), dim=-1))
                if labels is None:
                    member_labels.append(y)

            probs_list.append(torch.cat(member_probs, dim=0))
            if labels is None:
                labels = torch.cat(member_labels, dim=0)

        assert labels is not None
        mean_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # (N, C)
        n = labels.shape[0]
        nll = -torch.log(
            mean_probs[torch.arange(n, device=device), labels] + 1e-8
        ).mean()
        return nll.item()


def regression_mse_criterion(
    members: list[nn.Module],
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Default criterion for regression tasks.
    Ensemble prediction = mean of per-member raw outputs.
    Score = mean squared error over val_loader.
    members: list of trained nn.Module in eval() mode.
    Assumes model output shape (N, *) and target shape (N, *) are compatible
    with torch.nn.functional.mse_loss.
    """
    with torch.no_grad():
        preds_list: list[torch.Tensor] = []
        targets: torch.Tensor | None = None

        for model in members:
            model.to(device)
            model.eval()
            member_preds: list[torch.Tensor] = []
            member_targets: list[torch.Tensor] = []

            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                member_preds.append(model(x))
                if targets is None:
                    member_targets.append(y)

            preds_list.append(torch.cat(member_preds, dim=0))
            if targets is None:
                targets = torch.cat(member_targets, dim=0)

        assert targets is not None
        mean_preds = torch.stack(preds_list, dim=0).mean(dim=0)  # (N, *)
        return F.mse_loss(mean_preds, targets.float()).item()


def forward_select(
    pool: list[nn.Module],
    val_loader: DataLoader,
    ensemble_size: int,
    device: torch.device,
    criterion: Callable[[list[nn.Module], DataLoader, torch.device], float],
) -> list[nn.Module]:
    """
    Greedy forward stepwise selection without replacement (Section 4, NES paper).

    Selects `ensemble_size` models from `pool` that minimise the score returned
    by `criterion` on `val_loader`.

    Args:
        pool: Trained nn.Module instances.
        val_loader: Validation data loader yielding (inputs, labels/targets) batches.
        ensemble_size: Number of members to select (M in the paper).
        device: Device on which to run inference.
        criterion: Callable(members, val_loader, device) -> float.
            Must be lower-is-better. Called once per candidate at each greedy step.

    Returns:
        List of `ensemble_size` models selected from `pool`.
    """
    selected: list[nn.Module] = []

    for _ in range(ensemble_size):
        best_score = float("inf")
        best_model = None

        for candidate in pool:
            if candidate in selected:
                continue
            score = criterion(selected + [candidate], val_loader, device)
            if score < best_score:
                best_score = score
                best_model = candidate

        assert best_model is not None
        selected.append(best_model)

    return selected
