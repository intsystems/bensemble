from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def classification_nll_criterion(
    members: list[nn.Module],
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluates classification Negative Log-Likelihood of an ensemble candidate set.

    Predictions from all members are converted to softmax probabilities and averaged.

    Args:
        members: List of candidate neural network modules in eval mode.
        val_loader: DataLoader yielding validation (inputs, labels) batches.
        device: Device on which to run inference.

    Returns:
        float: Mean negative log-likelihood score over the validation set.
    """
    for model in members:
        model.to(device)
        model.eval()

    total_nll = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            mean_probs = torch.stack(
                [F.softmax(model(x), dim=-1) for model in members], dim=0
            ).mean(dim=0)
            picked = mean_probs[torch.arange(y.shape[0], device=device), y]
            total_nll += -torch.log(picked + 1e-8).sum().item()
            count += y.shape[0]

    return total_nll / count


def regression_mse_criterion(
    members: list[nn.Module],
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluates Mean Squared Error of an ensemble candidate set.

    Predictions from all members are averaged directly.

    Args:
        members: List of candidate neural network modules in eval mode.
        val_loader: DataLoader yielding validation (inputs, targets) batches.
        device: Device on which to run inference.

    Returns:
        float: Mean squared error score over the validation set.
    """
    for model in members:
        model.to(device)
        model.eval()

    total_se = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0].to(device), batch[1].to(device).float()
            mean_preds = torch.stack([model(x) for model in members], dim=0).mean(dim=0)
            total_se += F.mse_loss(mean_preds, y, reduction="sum").item()
            count += y.numel()

    return total_se / count


def forward_select(
    pool: list[nn.Module],
    val_loader: DataLoader,
    ensemble_size: int,
    device: torch.device,
    criterion: Callable[[list[nn.Module], DataLoader, torch.device], float],
) -> list[nn.Module]:
    """Performs greedy forward stepwise ensemble selection without replacement.

    Iteratively selects models from `pool` that minimize the metric returned
    by `criterion` on `val_loader`.

    Args:
        pool: Candidate neural network modules.
        val_loader: Validation DataLoader yielding (inputs, targets) batches.
        ensemble_size: Number of members to select.
        device: Device on which to execute evaluation.
        criterion: Evaluation callable `(members, val_loader, device) -> float`
            where lower values indicate better performance.

    Returns:
        list[nn.Module]: List of selected model instances of length `ensemble_size`.
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
