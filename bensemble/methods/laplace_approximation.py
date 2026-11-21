import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from bensemble.core.ensemble import Ensemble


class LaplaceApproximation:
    """Kronecker-factored Laplace approximation for neural networks."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: str = "regression",
        prior_precision: float = 1.0,
        damping: float = 1e-6,
        regularization: str = "exact",
        verbose: bool = False,
    ):
        """Initializes the LaplaceApproximation instance.

        Args:
            model: PyTorch model to approximate.
            likelihood: Task likelihood, either "classification" or "regression". Defaults to "regression".
            prior_precision: Prior precision hyperparameter (scalar tau). Defaults to 1.0.
            damping: Numerical stabilization term. In exact mode it is added to
                the posterior precision diagonal. Defaults to 1e-6.
            regularization: "exact" adds the prior to the Kronecker product
                of the curvature factors. "paper" and "legacy" regularize
                each factor separately.
            verbose: If True, prints progress details during computation. Defaults to False.

        Raises:
            ValueError: If likelihood or regularization values are unsupported.
        """
        if likelihood not in ["classification", "regression"]:
            raise ValueError(f"Unsupported likelihood: {likelihood}")
        if regularization not in ["exact", "legacy", "paper"]:
            raise ValueError(f"Unsupported regularization: {regularization}")
        if not math.isfinite(prior_precision) or prior_precision < 0:
            raise ValueError("prior_precision must be finite and non-negative.")
        if not math.isfinite(damping) or damping < 0:
            raise ValueError("damping must be finite and non-negative.")

        self.model = model
        self.is_fitted = False
        self.device = next(model.parameters()).device

        self.likelihood = likelihood
        self.prior_precision = float(prior_precision)
        self.damping = damping
        self.regularization = regularization
        self.verbose = verbose

        self.kronecker_factors: dict[str, dict[str, torch.Tensor]] = {}
        self.sampling_factors: dict[str, dict[str, Any]] = {}
        self.dataset_size = 1

        self.hook_handles = []
        self.activations: dict[str, torch.Tensor] = {}
        self._pre_activations: dict[str, torch.Tensor] = {}

    def toggle_verbose(self) -> None:
        """Toggles verbosity flag."""
        self.verbose = not self.verbose
        print("Verbose:", "on" if self.verbose else "off")

    def compute_curvature(
        self,
        train_loader: DataLoader,
        num_samples: int = 1000,
    ) -> None:
        """Estimates Kronecker factors of generalized Gauss-Newton curvature.

        Linear layers must receive 2D batches and be called once per forward
        pass. The model must process examples independently in eval mode.

        Args:
            train_loader: DataLoader yielding training data and targets.
            num_samples: Maximum number of samples to process for curvature estimation. Defaults to 1000.
        """
        self.is_fitted = False
        self.dataset_size = len(train_loader.dataset)

        if self.verbose:
            print("Registering hooks...")
        self._register_hooks()

        try:
            if self.verbose:
                print("Estimating Kronecker factors...")
            self._estimate_kronecker_factors(train_loader, num_samples)
        finally:
            if self.verbose:
                print("Removing hooks...")
            self._remove_hooks()

        self.is_fitted = True

        if self.verbose:
            print("Curvature computation completed!")

    def _register_hooks(self) -> None:
        """Records Linear inputs and pre-activations for curvature estimation."""
        self._remove_hooks()

        def make_forward_hook(layer_name):
            def forward_hook(module, inputs, output):
                if layer_name in self._pre_activations:
                    raise ValueError(
                        "Laplace curvature does not support reused Linear layers."
                    )
                x = inputs[0]
                if x.ndim != 2:
                    raise ValueError("Laplace curvature requires 2D Linear inputs.")
                self.activations[layer_name] = x.detach().clone()
                # Frozen layers still need derivatives with respect to their outputs.
                if not output.requires_grad:
                    output.requires_grad_()
                self._pre_activations[layer_name] = output
                # Preserve pre-activations when the next operation is in-place.
                return output.clone()

            return forward_hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(make_forward_hook(name))
                self.hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        """Removes all registered PyTorch hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.activations.clear()
        self._pre_activations.clear()

    def _compute_pre_activation_hessian(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the loss Hessian with respect to model pre-activations.

        Args:
            output: Model output logits/predictions of shape (batch_size, output_dim).
            target: Target labels or values.

        Returns:
            torch.Tensor: Hessian tensor of shape (batch_size, output_dim, output_dim).
        """
        batch_size = output.shape[0]

        if self.likelihood == "classification":
            probs = F.softmax(output, dim=1)
            diag_p = torch.diag_embed(probs)
            probs_outer = torch.einsum("bi,bj->bij", probs, probs)
            return diag_p - probs_outer

        output_dim = output.shape[1]
        return (
            torch.eye(output_dim, device=output.device, dtype=output.dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

    def _backward_hessian(
        self,
        output: torch.Tensor,
        hessian_final: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Computes mean J.T @ H @ J at each Linear pre-activation."""
        if not self._pre_activations:
            return {}

        pre_activations = tuple(self._pre_activations.values())
        jacobians: dict[str, list[torch.Tensor]] = {
            name: [] for name in self._pre_activations
        }
        for index in range(output.shape[1]):
            # Eval-mode examples are independent, so a summed output coordinate
            # gives each example's Jacobian in one backward pass.
            gradients = torch.autograd.grad(
                output[:, index].sum(),
                pre_activations,
                retain_graph=index < output.shape[1] - 1,
                allow_unused=True,
            )
            for (name, activation), gradient in zip(
                self._pre_activations.items(), gradients
            ):
                jacobians[name].append(
                    gradient.detach()
                    if gradient is not None
                    else torch.zeros_like(activation)
                )

        hessians = {}
        for name, gradients in jacobians.items():
            jacobian = torch.stack(gradients, dim=1)
            weighted = hessian_final @ jacobian
            # Sum over examples without allocating a batch of hidden x hidden matrices.
            curvature = jacobian.flatten(0, 1).T @ weighted.flatten(0, 1)
            hessians[name] = self._symmetrize(curvature / output.shape[0])
        return hessians

    def _estimate_kronecker_factors(
        self,
        train_loader: DataLoader,
        num_samples: int,
    ) -> None:
        """Estimates Kronecker curvature factors and matrix square roots.

        Args:
            train_loader: DataLoader providing training samples.
            num_samples: Maximum number of samples to process.
        """
        self.model.eval()

        accumulators: dict[str, dict[str, Any]] = {}
        sample_count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            if sample_count >= num_samples:
                break

            data = data.to(self.device)
            target = target.to(self.device)

            remaining = num_samples - sample_count
            if data.shape[0] > remaining:
                data = data[:remaining]
                target = target[:remaining]

            batch_size = data.shape[0]

            self.activations.clear()
            self._pre_activations.clear()
            with torch.inference_mode(False), torch.enable_grad():
                if data.is_inference():
                    data = data.clone()
                output = self.model(data)
                h_final = self._compute_pre_activation_hessian(output.detach(), target)
                layer_hessians = self._backward_hessian(output, h_final)
            self._pre_activations.clear()

            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if (
                        isinstance(module, nn.Linear)
                        and name in self.activations
                        and name in layer_hessians
                    ):
                        a = self.activations[name]
                        h = layer_hessians[name]

                        if name not in accumulators:
                            in_dim = a.shape[1]
                            out_dim = h.shape[0]

                            accumulators[name] = {
                                "Q_sum": torch.zeros(
                                    in_dim, in_dim, device=self.device, dtype=a.dtype
                                ),
                                "H_sum": torch.zeros(
                                    out_dim, out_dim, device=self.device, dtype=h.dtype
                                ),
                                "count": 0,
                                "in_dim": in_dim,
                                "out_dim": out_dim,
                            }

                        batch_q = torch.einsum("bi,bj->ij", a, a) / batch_size
                        batch_h = h.mean(0) if h.dim() == 3 else h

                        accumulators[name]["Q_sum"] += batch_q * batch_size
                        accumulators[name]["H_sum"] += batch_h * batch_size
                        accumulators[name]["count"] += batch_size

            sample_count += batch_size

            if sample_count % 1000 == 0 and self.verbose:
                print(f"Processed {sample_count} samples...")

        with torch.no_grad():
            self.kronecker_factors.clear()
            self.sampling_factors.clear()

            for name, acc in accumulators.items():
                if acc["count"] == 0:
                    continue

                q = acc["Q_sum"] / acc["count"]
                h = acc["H_sum"] / acc["count"]

                q = self._symmetrize(q)
                h = self._symmetrize(h)

                n = float(self.dataset_size)
                tau = float(self.prior_precision)

                if self.regularization == "exact":
                    q_eig, q_basis = torch.linalg.eigh(q)
                    h_eig, h_basis = torch.linalg.eigh(h)
                    if tau == 0 and self.damping == 0:
                        for eig in (q_eig, h_eig):
                            tol = torch.finfo(eig.dtype).eps * eig.numel() * eig[-1]
                            if eig[0] <= tol:
                                raise ValueError(
                                    "Singular Laplace precision. Use positive prior_precision or damping."
                                )
                    # Both factors are PSD up to roundoff. In their joint basis
                    # the precision is diagonal, with one entry per weight.
                    precision = n * torch.outer(h_eig.clamp_min(0), q_eig.clamp_min(0))
                    precision += tau + self.damping
                    self.kronecker_factors[name] = {"Q": q, "H": h}
                    self.sampling_factors[name] = {
                        "input_basis": q_basis,
                        "output_basis": h_basis,
                        "posterior_std": precision.rsqrt(),
                        "weight_shape": (acc["out_dim"], acc["in_dim"]),
                    }
                    continue

                eye_q = torch.eye(acc["in_dim"], device=self.device, dtype=q.dtype)
                eye_h = torch.eye(acc["out_dim"], device=self.device, dtype=h.dtype)

                if self.regularization == "legacy":
                    q_reg = n * q + tau * eye_q
                    h_reg = n * h + tau * eye_h
                else:
                    q_reg = (n**0.5) * q + (tau**0.5) * eye_q
                    h_reg = (n**0.5) * h + (tau**0.5) * eye_h

                q_reg = self._stabilize_spd(q_reg)
                h_reg = self._stabilize_spd(h_reg)

                self.kronecker_factors[name] = {
                    "Q": q_reg.detach().clone(),
                    "H": h_reg.detach().clone(),
                }

                q_cov = torch.linalg.inv(q_reg)
                h_cov = torch.linalg.inv(h_reg)

                l_q = self._matrix_sqrt(q_cov)
                l_h = self._matrix_sqrt(h_cov)

                self.sampling_factors[name] = {
                    "L_U": l_q,
                    "L_V": l_h,
                    "weight_shape": (acc["out_dim"], acc["in_dim"]),
                }

    def _matrix_sqrt(self, matrix: torch.Tensor) -> torch.Tensor:
        """Computes the symmetric positive semi-definite matrix square root.

        Args:
            matrix: Input square matrix tensor.

        Returns:
            torch.Tensor: Matrix square root.
        """
        matrix = self._stabilize_spd(self._symmetrize(matrix))
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = eigvals.clamp_min(self.damping)
        return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

    def sample_models(
        self, n_models: int = 10, temperature: float = 1.0
    ) -> list[nn.Module]:
        """Samples model parameters from the approximated Gaussian posterior.

        Args:
            n_models: Number of models to sample. Defaults to 10.
            temperature: Sampling temperature scaling factor. Defaults to 1.0.

        Returns:
            list[nn.Module]: Sampled model instances in eval mode.

        Raises:
            RuntimeError: If curvature has not been computed prior to sampling.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Laplace curvature not computed. Call compute_curvature() first."
            )

        samples = []
        modules = dict(self.model.named_modules())

        for _ in range(n_models):
            sampled_state = copy.deepcopy(self.model.state_dict())

            for name, factors in self.sampling_factors.items():
                module = modules[name]

                mean_weight = module.weight.detach()
                weight_shape = factors["weight_shape"]
                z = torch.randn(
                    weight_shape, device=self.device, dtype=mean_weight.dtype
                )
                if "posterior_std" in factors:
                    z = z * factors["posterior_std"].to(mean_weight)
                    left, right = factors["output_basis"], factors["input_basis"]
                else:
                    left, right = factors["L_V"], factors["L_U"]
                delta = left.to(mean_weight) @ z @ right.to(mean_weight).T
                sampled_weight = mean_weight + temperature * delta

                prefix = f"{name}." if name else ""
                sampled_state[f"{prefix}weight"] = sampled_weight.detach().cpu()

                if module.bias is not None:
                    sampled_state[f"{prefix}bias"] = module.bias.detach().cpu()

            model_sample = copy.deepcopy(self.model)
            model_sample.load_state_dict(sampled_state, strict=True)
            model_sample.to(self.device)
            model_sample.eval()
            samples.append(model_sample)

        return samples

    def build_ensemble(self, n_members: int = 10, temperature: float = 1.0) -> Ensemble:
        """Builds an Ensemble module sampled from the Laplace posterior.

        Args:
            n_members: Number of ensemble members to draw. Defaults to 10.
            temperature: Sampling temperature for posterior weights. Defaults to 1.0.

        Returns:
            Ensemble: Ensemble instance wrapping the sampled models.
        """
        return Ensemble.from_posterior(
            self, n_members=n_members, temperature=temperature
        )

    def _get_ensemble_state(self) -> dict[str, Any]:
        """Serializes current approximation state into a dictionary.

        Returns:
            dict[str, Any]: Dictionary containing approximation parameters and state.
        """
        return {
            "model_state": self.model.state_dict(),
            "is_fitted": self.is_fitted,
            "likelihood": self.likelihood,
            "sampling_factors": self.sampling_factors,
            "dataset_size": self.dataset_size,
            "prior_precision": self.prior_precision,
            "regularization": self.regularization,
            "damping": self.damping,
        }

    def _set_ensemble_state(self, state: dict[str, Any]) -> None:
        """Restores approximation state from a dictionary.

        Args:
            state: Dictionary containing previously saved state.
        """
        self.model.load_state_dict(state["model_state"])
        self.is_fitted = state["is_fitted"]
        self.likelihood = state["likelihood"]
        self.sampling_factors = state["sampling_factors"]
        self.dataset_size = state["dataset_size"]
        self.prior_precision = state["prior_precision"]
        self.regularization = state.get("regularization", self.regularization)
        self.damping = state.get("damping", self.damping)
        self.hook_handles = []

    def _stabilize_spd(self, matrix: torch.Tensor) -> torch.Tensor:
        """Adds damping to diagonal elements to ensure symmetric positive-definiteness.

        Args:
            matrix: Input square matrix tensor.

        Returns:
            torch.Tensor: Damped symmetric positive-definite matrix.
        """
        eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        return self._symmetrize(matrix) + self.damping * eye

    @staticmethod
    def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
        """Ensures matrix symmetry: 0.5 * (M + M^T).

        Args:
            matrix: Input square matrix tensor.

        Returns:
            torch.Tensor: Symmetrized matrix.
        """
        return 0.5 * (matrix + matrix.T)
