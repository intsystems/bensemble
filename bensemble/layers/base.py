import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseBayesianLayer(nn.Module):
    """
    Base class for all bayesian layers.

    Computes KL-divergence automatically for all parameters ending with `_mu` and `_rho`.
    """

    def __init__(self, prior_sigma: float = 1.0):
        super().__init__()
        self.prior_sigma = prior_sigma

    def kl_divergence(self) -> torch.Tensor:
        """
        Computes KL-divergence KL(q || p) for all bayesian weights of the layer.
        p(w) = N(0, prior_sigma^2)
        q(w) = N(mu, sigma^2), where sigma = softplus(rho)
        """
        total_kl = 0.0

        for name, param in self.named_parameters():
            if name.endswith("_mu"):
                rho_name = name.replace("_mu", "_rho")

                if hasattr(self, rho_name):
                    mu = param
                    rho = getattr(self, rho_name)

                    total_kl += self._compute_kl_for_param(mu, rho)

        return total_kl

    def get_pruning_masks(self, threshold: float = 0.83) -> dict:
        """Returns binary masks for parameters satisfying the SNR threshold.

        Implements Graves' pruning heuristic where weights with low
        Signal-to-Noise Ratio are considered redundant and can be removed.

        Args:
            threshold (float, optional): The SNR threshold (|mu|/sigma).
                Defaults to 0.83, the "safe" threshold suggested by Graves.

        Returns:
            dict[str, torch.Tensor]: A dictionary mapping parameter names to
                binary masks (1.0 for keeping, 0.0 for pruning).
        """
        snr_dict = self._get_snr_dict()
        return {name: (val > threshold).float() for name, val in snr_dict.items()}

    def apply_pruning(self, threshold: float = 0.83) -> float:
        """
        Applies pruning in-place: zeros out the means and minimizes the variance
        for weights that fall below the SNR threshold.

        Returns:
            float: Sparsity of the layer (percentage of pruned weights, 0.0 to 1.0).
        """
        masks = self.get_pruning_masks(threshold)

        total_weights = 0
        pruned_weights = 0

        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith("_mu"):
                    mask = masks[name]

                    total_weights += mask.numel()
                    pruned_weights += (mask == 0.0).sum().item()
                    param.data *= mask

                    rho_name = name.replace("_mu", "_rho")
                    if hasattr(self, rho_name):
                        rho = getattr(self, rho_name)
                        rho.data = torch.where(
                            mask.bool(),
                            rho.data,
                            torch.tensor(-1e8, device=rho.device, dtype=rho.dtype),
                        )

        sparsity = pruned_weights / total_weights if total_weights > 0 else 0.0
        return sparsity

    def _get_snr_dict(self) -> dict[str, torch.Tensor]:
        """Computes the Signal-to-Noise Ratio (SNR) for all Bayesian parameters.

        SNR is defined as the absolute mean of the weight divided by its
        standard deviation: SNR = |mu| / sigma.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing SNR tensors
                for each Bayesian parameter in the layer.
        """
        snr_dict = {}

        for name, param in self.named_parameters():
            if not name.endswith("_mu"):
                continue

            mu = param

            rho_name = name.replace("_mu", "_rho")
            rho = getattr(self, rho_name)

            sigma = F.softplus(rho)

            snr_dict[name] = torch.abs(mu) / sigma

        return snr_dict

    def _compute_kl_for_param(
        self, mu: torch.Tensor, rho: torch.Tensor
    ) -> torch.Tensor:
        """Computes KL-divergence for pair (mu, rho)."""
        sigma = F.softplus(rho)

        num_el = mu.numel()

        const_term = -0.5 * num_el
        log_prior_term = math.log(self.prior_sigma) * num_el

        kl = -torch.log(sigma).sum() + (sigma**2 + mu**2).sum() / (
            2 * self.prior_sigma**2
        )

        return kl + log_prior_term + const_term
