import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..core.base import BaseBayesianEnsemble


class LaplaceApproximation(BaseBayesianEnsemble):
    """
    Kronecker-factored Laplace approximation for neural networks.

    Backwards-compatible with the old API:
        fit(...)
        compute_posterior(...)
        sample_models(n_models=...)
        predict(...)

    Also compatible with posterior-source style APIs:
        sample_models(n_members=...)
    """

    def __init__(
        self,
        model: nn.Module,
        pretrained: bool = True,
        likelihood: str = "classification",
        verbose: bool = False,
        damping: float = 1e-6,
        regularization: str = "legacy",
    ):
        super().__init__(model)

        if likelihood not in ["classification", "regression"]:
            raise ValueError(f"Unsupported likelihood: {likelihood}")
        if regularization not in ["legacy", "paper"]:
            raise ValueError(f"Unsupported regularization: {regularization}")

        self.model = model
        self.is_fitted = False
        self.device = next(model.parameters()).device

        self.likelihood = likelihood
        self.pretrained = pretrained
        self.verbose = verbose
        self.damping = damping
        self.regularization = regularization

        self.kronecker_factors: Dict[str, Dict[str, torch.Tensor]] = {}
        self.sampling_factors: Dict[str, Dict[str, Any]] = {}
        self.dataset_size = 1
        self.prior_precision = 1.0

        self.hook_handles = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.pre_activation_hessians: Dict[str, torch.Tensor] = {}

    def toggle_verbose(self):
        self.verbose = not self.verbose
        print("Verbose:", "on" if self.verbose else "off")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        lr: float = 1e-3,
        prior_precision: float = 1.0,
        num_samples: int = 1000,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {}

        if not self.pretrained:
            if self.verbose:
                print("Training model...")

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            history["train_loss"] = []

            for epoch in range(num_epochs):
                self.model.train()
                train_loss = 0.0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad(set_to_none=True)
                    output = self.model(batch_x)

                    if self.likelihood == "classification":
                        loss = F.cross_entropy(output, batch_y.long())
                    else:
                        batch_y = batch_y.to(output.dtype)
                        if batch_y.shape != output.shape:
                            batch_y = batch_y.view_as(output)
                        loss = F.mse_loss(output, batch_y)

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                total_loss = train_loss / max(1, len(train_loader))
                history["train_loss"].append(total_loss)

                if self.verbose:
                    print(f"Epoch {epoch}: training loss = {total_loss:.4f}")

            self.pretrained = True

        self.compute_posterior(train_loader, prior_precision, num_samples)
        return history

    def compute_posterior(
        self,
        train_loader: DataLoader,
        prior_precision: float = 1.0,
        num_samples: int = 1000,
    ) -> None:
        self.prior_precision = float(prior_precision)
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
            print("Posterior computation completed!")

    def _register_hooks(self) -> None:
        self._remove_hooks()
        self.activations = {}
        self.pre_activation_hessians = {}

        def make_forward_hook(layer_name):
            def forward_hook(module, inputs, output):
                x = inputs[0] if isinstance(inputs, tuple) else inputs
                if x.dim() > 2:
                    x = x.flatten(start_dim=1)
                self.activations[layer_name] = x.detach()
            return forward_hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(make_forward_hook(name))
                self.hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def _compute_pre_activation_hessian(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
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
        hessian_final: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hessians: Dict[str, torch.Tensor] = {}

        linear_layers = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
        ]

        if len(linear_layers) == 0:
            return hessians

        current_hessian = (
            hessian_final.mean(0)
            if hessian_final.dim() == 3
            else hessian_final
        )

        final_layer_name, _ = linear_layers[-1]
        hessians[final_layer_name] = current_hessian

        for i in range(len(linear_layers) - 2, -1, -1):
            current_layer_name, _ = linear_layers[i]
            _, next_layer = linear_layers[i + 1]

            w_next = next_layer.weight.detach()
            current_hessian = w_next.T @ current_hessian @ w_next
            current_hessian = self._symmetrize(current_hessian)

            hessians[current_layer_name] = current_hessian

        return hessians

    def _estimate_kronecker_factors(
        self,
        train_loader: DataLoader,
        num_samples: int,
    ) -> None:
        self.model.eval()

        accumulators: Dict[str, Dict[str, Any]] = {}
        sample_count = 0

        if self.verbose:
            print(f"Processing up to {num_samples} samples...")

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

            with torch.no_grad():
                output = self.model(data)
                h_final = self._compute_pre_activation_hessian(output, target)
                layer_hessians = self._backward_hessian(h_final)

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
                                    in_dim,
                                    in_dim,
                                    device=self.device,
                                    dtype=a.dtype,
                                ),
                                "H_sum": torch.zeros(
                                    out_dim,
                                    out_dim,
                                    device=self.device,
                                    dtype=h.dtype,
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

        if self.verbose:
            print("Computing final Kronecker factors...")

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

                eye_q = torch.eye(acc["in_dim"], device=self.device, dtype=q.dtype)
                eye_h = torch.eye(acc["out_dim"], device=self.device, dtype=h.dtype)

                if self.regularization == "legacy":
                    q_reg = n * q + tau * eye_q
                    h_reg = n * h + tau * eye_h
                else:
                    q_reg = (n ** 0.5) * q + (tau ** 0.5) * eye_q
                    h_reg = (n ** 0.5) * h + (tau ** 0.5) * eye_h

                q_reg = self._stabilize_spd(q_reg)
                h_reg = self._stabilize_spd(h_reg)

                self.kronecker_factors[name] = {
                    "Q": q_reg.detach().clone(),
                    "H": h_reg.detach().clone(),
                }

                if self.verbose:
                    print(f"Layer {name}:")
                    print(f"  Q shape: {q_reg.shape}, H shape: {h_reg.shape}")
                    print(f"  Q norm: {torch.norm(q_reg).item():.6f}")
                    print(f"  H norm: {torch.norm(h_reg).item():.6f}")
                    print(f"  cond(Q): {torch.linalg.cond(q_reg).item():.2e}")
                    print(f"  cond(H): {torch.linalg.cond(h_reg).item():.2e}")

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
        matrix = self._stabilize_spd(self._symmetrize(matrix))
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = eigvals.clamp_min(self.damping)
        return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

    def sample_models(
        self,
        n_models: int = 10,
        temperature: float = 1.0,
        n_members: Optional[int] = None,
    ) -> List[nn.Module]:
        if n_members is not None:
            n_models = n_members

        if not self.is_fitted:
            raise RuntimeError("LaplaceApproximation must be fitted before sampling.")

        samples = []
        modules = dict(self.model.named_modules())

        for _ in range(n_models):
            sampled_state = copy.deepcopy(self.model.state_dict())

            for name, factors in self.sampling_factors.items():
                module = modules[name]

                mean_weight = module.weight.detach()
                l_q = factors["L_U"].to(device=self.device, dtype=mean_weight.dtype)
                l_h = factors["L_V"].to(device=self.device, dtype=mean_weight.dtype)

                weight_shape = factors["weight_shape"]
                z = torch.randn(weight_shape, device=self.device, dtype=mean_weight.dtype)

                sampled_weight = mean_weight + temperature * (l_h @ z @ l_q.T)
                sampled_state[f"{name}.weight"] = sampled_weight.detach().cpu()

                if module.bias is not None:
                    sampled_state[f"{name}.bias"] = module.bias.detach().cpu()

            model_sample = copy.deepcopy(self.model)
            model_sample.load_state_dict(sampled_state, strict=True)
            model_sample.to(self.device)
            model_sample.eval()
            samples.append(model_sample)

        return samples

    def predict(
        self,
        X: torch.Tensor,
        n_samples: int = 100,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(self.device)
        predictions = []

        for sampled_model in self.sample_models(
            n_models=n_samples,
            temperature=temperature,
        ):
            sampled_model.eval()
            with torch.no_grad():
                predictions.append(sampled_model(X))

        predictions = torch.stack(predictions, dim=0)

        if self.likelihood == "classification":
            probs = F.softmax(predictions, dim=-1)
            mean_probs = probs.mean(dim=0)
            uncertainty = -(
                mean_probs * torch.log(mean_probs.clamp_min(1e-8))
            ).sum(dim=-1)
            return mean_probs, uncertainty

        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0, unbiased=False)
        return mean, variance

    def build_ensemble(self, n_members: int = 10, **kwargs):
        from ..core.ensemble import Ensemble

        return Ensemble.from_posterior(self, n_members=n_members, **kwargs)

    def _get_ensemble_state(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "is_fitted": self.is_fitted,
            "device": self.device,
            "likelihood": self.likelihood,
            "pretrained": self.pretrained,
            "kronecker_factors": self.kronecker_factors,
            "sampling_factors": self.sampling_factors,
            "dataset_size": self.dataset_size,
            "prior_precision": self.prior_precision,
            "damping": self.damping,
            "regularization": self.regularization,
            "verbose": self.verbose,
        }

    def _set_ensemble_state(self, state: Dict[str, Any]):
        if state["likelihood"] in ["classification", "regression"]:
            self.likelihood = state["likelihood"]
        else:
            raise ValueError(f"Unsupported likelihood: {state['likelihood']}")

        self.model = state["model"]
        self.is_fitted = state["is_fitted"]
        self.device = state["device"]
        self.pretrained = state["pretrained"]
        self.kronecker_factors = state["kronecker_factors"]
        self.sampling_factors = state["sampling_factors"]
        self.dataset_size = state["dataset_size"]
        self.prior_precision = state.get("prior_precision", 1.0)
        self.damping = state.get("damping", 1e-6)
        self.regularization = state.get("regularization", "legacy")
        self.verbose = state.get("verbose", False)
        self.hook_handles = []

    def _stabilize_spd(self, matrix: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        return self._symmetrize(matrix) + self.damping * eye

    @staticmethod
    def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
        return 0.5 * (matrix + matrix.T)