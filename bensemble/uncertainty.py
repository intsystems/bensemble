from typing import Tuple
import torch


def decompose_classification_uncertainty(
    probs: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decomposes total predictive uncertainty into aleatoric and epistemic components.

    Args:
        probs (torch.Tensor): Predicted probabilities from the ensemble.
            Expected shape:[M_models, Batch_size, Num_classes].
        eps (float, optional): Small epsilon for numerical stability in torch.log().
            Defaults to 1e-8.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - total_unc (torch.Tensor): Total uncertainty (entropy of the mean). Shape: [Batch].
            - aleatoric_unc (torch.Tensor): Aleatoric uncertainty (mean of entropies). Shape: [Batch].
            - epistemic_unc (torch.Tensor): Epistemic uncertainty (mutual information). Shape: [Batch].
    """
    mean_probs = probs.mean(dim=0)
    total_unc = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)
    entropies = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    aleatoric_unc = entropies.mean(dim=0)
    epistemic_unc = total_unc - aleatoric_unc

    return total_unc, aleatoric_unc, epistemic_unc
