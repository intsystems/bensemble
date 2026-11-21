import math
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

    Scores the arithmetic mean of the members' class probabilities in log space.

    Args:
        members: List of candidate neural network modules in eval mode.
        val_loader: DataLoader yielding validation (inputs, labels) batches,
            with class indices shaped (batch,) or (batch, 1).
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
            logits = torch.stack([model(x) for model in members])
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            if y.ndim == 2:
                y = y.squeeze(1)
            if y.shape != (logits.size(1),):
                raise ValueError("Expected one class index per input.")
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs[:, torch.arange(y.size(0), device=device), y]
            log_mixture = torch.logsumexp(target_log_probs, dim=0) - math.log(
                len(members)
            )
            total_nll -= log_mixture.sum().item()
            count += y.shape[0]

    return total_nll / count


def regression_mse_criterion(
    members: list[nn.Module],
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluates Mean Squared Error of an ensemble candidate set.

    Predictions from all members are averaged directly. Scalar predictions
    and targets can each have shape (batch,) or (batch, 1). For multiple
    outputs, prediction and target shapes must match.

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
            if mean_preds.shape != y.shape:
                if y.ndim == 1 and mean_preds.shape == (y.size(0), 1):
                    y = y.unsqueeze(-1)
                elif mean_preds.ndim == 1 and y.shape == (mean_preds.size(0), 1):
                    y = y.squeeze(-1)
                else:
                    raise ValueError(
                        f"Regression prediction shape {tuple(mean_preds.shape)} "
                        f"does not match target shape {tuple(y.shape)}."
                    )
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
