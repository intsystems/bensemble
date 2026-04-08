from typing import Dict
import torch
import torch.nn.functional as F


def negative_log_likelihood(
    probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> float:
    """
    Computes the Negative Log-Likelihood (NLL) for predicted probabilities.

    This is a strictly proper scoring rule. Lower is better.

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [Batch, Num_classes].
        targets (torch.Tensor): Ground truth class indices of shape[Batch].
        eps (float, optional): Small value to prevent log(0). Defaults to 1e-8.

    Returns:
        float: The average NLL over the batch.
    """
    log_probs = torch.log(probs + eps)
    loss = F.nll_loss(log_probs, targets)
    return loss.item()


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes the Brier Score for multi-class classification.

    The Brier Score is the mean squared difference between the predicted
    probability distribution and the one-hot encoded true label.
    Lower is better.

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [Batch, Num_classes].
        targets (torch.Tensor): Ground truth class indices of shape [Batch].

    Returns:
        float: The Brier score.
    """
    num_classes = probs.shape[-1]
    one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()

    squared_diff = (probs - one_hot_targets) ** 2
    score = torch.mean(torch.sum(squared_diff, dim=1))

    return score.item()


def expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> float:
    """
    Computes the Expected Calibration Error (ECE).

    Divides the confidence space into `n_bins` and measures the weighted
    absolute difference between the model's accuracy and confidence in each bin.
    Lower is better (0.0 means perfectly calibrated).

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [Batch, Num_classes].
        targets (torch.Tensor): Ground truth class indices of shape [Batch].
        n_bins (int, optional): Number of bins. Defaults to 15.

    Returns:
        float: The ECE score.
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece.item()


def reliability_diagram(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> Dict[str, list]:
    """
    Computes data points needed to plot a Reliability Diagram.

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [Batch, Num_classes].
        targets (torch.Tensor): Ground truth class indices of shape [Batch].
        n_bins (int, optional): Number of bins. Defaults to 15.

    Returns:
        Dict[str, list]: A dictionary containing lists of:
            - 'confidences': Average confidence in each bin.
            - 'accuracies': Average accuracy in each bin.
            - 'proportions': Fraction of samples in each bin.
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    results = {"confidences": [], "accuracies": [], "proportions": []}

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0.0:
            results["accuracies"].append(accuracies[in_bin].float().mean().item())
            results["confidences"].append(confidences[in_bin].mean().item())
            results["proportions"].append(prop_in_bin)
        else:
            results["accuracies"].append(0.0)
            results["confidences"].append((bin_lower.item() + bin_upper.item()) / 2.0)
            results["proportions"].append(0.0)

    return results
