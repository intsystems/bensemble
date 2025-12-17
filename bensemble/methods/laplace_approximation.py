import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..core.base import BaseBayesianEnsemble


class LaplaceApproximation(BaseBayesianEnsemble):
    """
    A scalable Laplace approximation for neural networks based on Kronecker-factored curvature.
    """

    def __init__(
        self,
        model: nn.Module,
        pretrained: bool = True,
        likelihood: str = "classification",
        verbose: bool = False,
    ):
        self.model = model
        self.is_fitted = False
        self.device = next(model.parameters()).device
        if likelihood in ["classification", "regression"]:
            self.likelihood = likelihood
        else:
            raise ValueError(f"Unsupported likelihood: {likelihood}")

        self.pretrained = pretrained

        # Storage for Kronecker factors
        self.kronecker_factors = {}
        self.sampling_factors = {}
        self.dataset_size = 1

        # Hook handles
        self.hook_handles = []

        self.verbose = verbose

    def toggle_verbose(self):
        """
        Turn verbose on or off.
        """
        self.verbose = not self.verbose
        print("Verbose:", "on" if self.verbose else "off")

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 100,
        lr: float = 1e-3,
        prior_precision: float = 1.0,
        num_samples: int = 1000,
    ) -> Dict[str, List[float]]:
        history = {}

        if not self.pretrained:
            if self.verbose:
                print("Training model...")
            # First train model for MAP estimation
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            history["train_loss"] = []

            for epoch in range(num_epochs):
                self.model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    output = self.model(batch_X)
                    if self.likelihood == "classification":
                        loss = F.cross_entropy(output, batch_y)
                    else:
                        loss = F.mse_loss(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                total_loss = train_loss / len(train_loader)
                history["train_loss"].append(total_loss)

                if self.verbose:
                    print(f"Epoch {epoch}: training loss = {total_loss:.4f}")

            self.pretrained = True

        self.compute_posterior(train_loader, prior_precision, num_samples)

        return history

    def compute_posterior(
        self,
        train_loader: DataLoader,
        prior_precision: float = 0.0,
        num_samples: int = 1000,
    ) -> None:
        """
        Compute the Kronecker-factored Laplace approximation.
        """
        self.prior_precision = prior_precision
        self.dataset_size = len(train_loader.dataset)

        if self.verbose:
            print("Registering hooks...")
        self._register_hooks()

        if self.verbose:
            print("Estimating Kronecker factors...")
        self._estimate_kronecker_factors(train_loader, num_samples)

        if self.verbose:
            print("Removing hooks...")
        self._remove_hooks()

        print("Posterior computation completed!")

    def _register_hooks(self) -> None:
        """
        Register forward hooks to capture activations and pre-activation Hessians.
        
        Hooks are registered on all linear layers to store input activations
        for computing the Q factor (covariance of input activations) in the
        Kronecker factorization.
        """
        self.activations = {}
        self.pre_activation_hessians = {}

        def make_forward_hook(layer_name):
            def forward_hook(module, input, output):
                if isinstance(input, tuple):
                    input = input[0]
                # Store input activations for Q factor
                self.activations[layer_name] = input.detach().clone()

            return forward_hook

        # Register hooks for linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                forward_handle = module.register_forward_hook(make_forward_hook(name))
                self.hook_handles.append(forward_handle)

    def _remove_hooks(self) -> None:
        """
        Remove all registered hooks.
        
        Clears the hook handles list to ensure no hooks remain attached
        after curvature estimation is complete.
        """
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def _compute_pre_activation_hessian(self, output, target):
        """
        Compute the Hessian with respect to the final layer pre-activations.
        
        This depends on the likelihood function:
        - For classification: Hessian = diag(p) - pp^T where p is softmax probability
        - For regression: Hessian = identity matrix
        
        Args:
            output: Model predictions (batch_size, output_dim)
            target: Ground truth labels/targets
            
        Returns:
            Pre-activation Hessian tensor of shape (batch_size, output_dim, output_dim) 
            for classification or (batch_size, output_dim, output_dim) for regression
        """
        batch_size = output.shape[0]

        if self.likelihood == "classification":
            # For cross-entropy loss with softmax, the Hessian is:
            # H = diag(p) - pp^T, where p is the softmax probabilities
            probs = F.softmax(output, dim=1)
            eye = torch.eye(probs.size(1), device=self.device).unsqueeze(0)  # (1, C, C)
            probs_outer = torch.einsum("bi,bj->bij", probs, probs)  # (B, C, C)
            hessian = (
                probs_outer - eye
            )  # This is actually -H, but we'll account for sign later
            return -hessian  # Return the negative Hessian of NLL

        else:  # self.likelihood == 'regression':
            # For MSE loss, the Hessian is identity
            output_dim = output.shape[1]
            hessian = (
                torch.eye(output_dim, device=self.device)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
            return hessian

    def _backward_hessian(self, hessian_final):
        """
        Recursively backpropagate the pre-activation Hessian through all layers.
        
        Uses weights of the NEXT layer to compute Hessian for the CURRENT layer
        according to the recursive formula: H_λ = W_{λ+1}^T @ H_{λ+1} @ W_{λ+1} + D_λ
        
        For piecewise linear activations (ReLU), D_λ ≈ 0.
        
        Args:
            hessian_final: Hessian for the final layer
            
        Returns:
            Dictionary mapping layer names to their computed Hessian matrices
        """
        hessians = {}

        # Get all linear layers in forward order
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))

        # Initialize with the final layer Hessian
        if hessian_final.dim() == 3:
            current_hessian = hessian_final.mean(0)  # Average over batch
        else:
            current_hessian = hessian_final

        final_layer_name, final_layer = linear_layers[-1]
        hessians[final_layer_name] = current_hessian

        # Backpropagate through layers in REVERSE order (from output to input)
        for i in range(
            len(linear_layers) - 2, -1, -1
        ):  # Start from second-to-last layer
            current_layer_name, current_layer = linear_layers[i]
            next_layer_name, next_layer = linear_layers[i + 1]

            # The recursive formula: H_λ = W_{λ+1}^T @ H_{λ+1} @ W_{λ+1} + D_λ
            # For piecewise linear activations (ReLU), D_λ ≈ 0
            W_next = next_layer.weight  # Shape: (out_dim_{λ+1}, in_dim_{λ+1})

            # H_current = W_next^T @ H_next @ W_next
            H_current = W_next.T @ current_hessian @ W_next

            # Store the Hessian for this layer
            hessians[current_layer_name] = H_current

            # Update for next iteration
            current_hessian = H_current

        return hessians

    def _estimate_kronecker_factors(
        self, train_loader: DataLoader, num_samples: int
    ) -> None:
        """
        Estimate Kronecker factors using proper Hessian computation.
        
        Processes training data to compute:
        1. Q factors: covariance of input activations
        2. H factors: pre-activation Hessians
        
        These are then regularized with prior precision and scaled by dataset size
        to form the posterior precision matrices in Kronecker-factored form.
        
        Args:
            train_loader: DataLoader providing training data
            num_samples: Maximum number of samples to use for estimation
        """
        self.model.eval()  # We want deterministic behavior for curvature estimation

        accumulators = {}
        sample_count = 0

        if self.verbose:
            print(f"Processing up to {num_samples} samples...")

        for batch_idx, (data, target) in enumerate(train_loader):
            if sample_count >= num_samples:
                break

            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]

            # Forward pass
            output = self.model(data)

            # Compute pre-activation Hessian for the final layer
            H_final = self._compute_pre_activation_hessian(output, target)

            # Backpropagate Hessian through all layers
            layer_hessians = self._backward_hessian(H_final)

            # Process each layer
            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if (
                        isinstance(module, nn.Linear)
                        and name in self.activations
                        and name in layer_hessians
                    ):
                        a = self.activations[name]  # input activations (B, in_dim)
                        H = layer_hessians[
                            name
                        ]  # pre-activation Hessian (out_dim, out_dim)

                        if name not in accumulators:
                            in_dim = a.shape[1]
                            out_dim = H.shape[0] if H.dim() == 2 else H.shape[1]

                            accumulators[name] = {
                                "Q_sum": torch.zeros(
                                    in_dim, in_dim, device=self.device
                                ),
                                "H_sum": torch.zeros(
                                    out_dim, out_dim, device=self.device
                                ),
                                "count": 0,
                                "in_dim": in_dim,
                                "out_dim": out_dim,
                            }

                        # Compute Q factor: covariance of input activations
                        batch_Q = torch.einsum("bi,bj->ij", a, a) / batch_size

                        # H factor is already computed from backpropagation
                        # Average if we have multiple samples
                        if H.dim() == 3:  # batch of Hessians
                            batch_H = H.mean(0)
                        else:
                            batch_H = H

                        accumulators[name]["Q_sum"] += batch_Q * batch_size
                        accumulators[name]["H_sum"] += batch_H * batch_size
                        accumulators[name]["count"] += batch_size

                sample_count += batch_size
                if sample_count % 1000 == 0 and self.verbose:
                    print(f"Processed {sample_count} samples...")

        # Compute final factors with proper regularization
        if self.verbose:
            print("Computing final Kronecker factors...")
        with torch.no_grad():
            for name, acc in accumulators.items():
                if acc["count"] == 0:
                    continue

                # Expected Kronecker factors (Equation 7 in the paper)
                Q = acc["Q_sum"] / acc["count"]  # E[Q_λ]
                H = acc["H_sum"] / acc["count"]  # E[H_λ]

                # Add prior precision and scale by dataset size (Equation 9)
                N = self.dataset_size
                tau = self.prior_precision

                # Posterior precision in the Kronecker-factored form:
                # tau * I + N * (Q ⊗ H) -> factors become (N * Q + tau * I) and (N * H + tau * I)
                Q_reg = N * Q + tau * torch.eye(acc["in_dim"], device=self.device)
                H_reg = N * H + tau * torch.eye(acc["out_dim"], device=self.device)

                # Store the precision matrices for sampling
                self.kronecker_factors[name] = {
                    "Q": Q_reg,  # Precision for rows
                    "H": H_reg,  # Precision for columns
                }

                if self.verbose:
                    print(
                        f"Layer {name}:\nQ shape {Q_reg.shape}, H shape {H_reg.shape}"
                    )
                    Q_norm = torch.norm(Q_reg).item()
                    H_norm = torch.norm(H_reg).item()
                    print(f"  Q norm: {Q_norm:.6f}, H norm: {H_norm:.6f}")
                    cond_Q = torch.linalg.cond(Q_reg)
                    cond_H = torch.linalg.cond(H_reg)
                    print(f"cond(Q)={cond_Q:.2e}, cond(H)={cond_H:.2e}")

                # Convert to covariance matrices for sampling
                U = torch.linalg.inv(Q_reg)  # Row covariance
                V = torch.linalg.inv(H_reg)  # Column covariance

                # Matrix sqrt for sampling
                L_U = self._matrix_sqrt(U)
                L_V = self._matrix_sqrt(V)

                self.sampling_factors[name] = {
                    "L_U": L_U,
                    "L_V": L_V,
                    "weight_shape": (acc["out_dim"], acc["in_dim"]),
                }

    def _matrix_sqrt(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute matrix square root using eigen decomposition.
        
        For a symmetric positive definite matrix A, computes L such that L @ L.T = A.
        
        Args:
            A: Symmetric positive definite matrix
            
        Returns:
            Matrix square root L where L @ L.T = A
        """
        # Eigen decomposition for symmetric matrices
        L, V = torch.linalg.eigh(A)
        return V @ torch.diag(torch.sqrt(L)) @ V.T

    def sample_models(
        self, n_models: int = 10, temperature: float = 1.0
    ) -> List[nn.Module]:
        """
        Sample weight matrices from the matrix normal posterior.
        
        Samples models from the approximate posterior distribution using the
        Kronecker-factored covariance structure. Each sample is generated as:
        W_sample = M + temperature * L_V @ Z @ L_U.T
        where M is the MAP estimate, L_U and L_V are matrix square roots of
        the Kronecker factors, and Z is standard normal noise.
        
        Args:
            n_models: Number of model samples to generate
            temperature: Scaling factor for the noise (higher = more exploration)
            
        Returns:
            List of sampled neural network models with different weight configurations
        """
        samples = []

        for i in range(n_models):
            weight_sample = {}

            for name, factors in self.sampling_factors.items():
                module = dict(self.model.named_modules())[name]
                M = module.weight.data  # MAP estimate

                L_V = factors["L_V"]  # Row precision (in_dim, in_dim)
                L_U = factors["L_U"]  # Column precision (out_dim, out_dim)
                weight_shape = factors["weight_shape"]

                # Generate single sample for this layer
                Z = torch.randn(weight_shape, device=self.device)
                W_sample = M + temperature * L_V @ Z @ L_U.T

                # Store in the model sample dictionary
                weight_sample[f"{name}.weight"] = W_sample.cpu()

            # Also include bias terms if they exist
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and module.bias is not None:
                    # For bias, we can use a simple diagonal approximation
                    # or sample from the appropriate marginal distribution
                    if name in self.kronecker_factors:
                        # The bias is typically part of the same layer's distribution
                        # but for simplicity, we'll use the MAP estimate for bias
                        weight_sample[f"{name}.bias"] = module.bias.data.clone()

            model_sample = copy.deepcopy(self.model)
            model_sample.load_state_dict(weight_sample)

            samples.append(model_sample)

        return samples

    def predict(
        self, X: torch.Tensor, n_samples: int = 100, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute predictive distribution using Monte Carlo sampling.
        
        For classification: returns mean class probabilities and predictive entropy
        For regression: returns mean prediction and predictive variance
        
        Args:
            X: Input tensor of shape (batch_size, input_features)
            n_samples: Number of Monte Carlo samples to draw
            temperature: Temperature scaling for posterior sampling
            
        Returns:
            Tuple of (predictive mean, predictive uncertainty):
            - For classification: (mean_probs, uncertainty)
            - For regression: (mean, variance)
        """
        predictions = []

        for _ in range(n_samples):
            sampled_model = self.sample_models(n_models=1, temperature=temperature)[0]
            sampled_model.to(self.device)

            with torch.no_grad():
                output = sampled_model(X)
                predictions.append(output)

            sampled_model.cpu()

        predictions = torch.stack(predictions)

        if self.likelihood == "classification":
            probs = F.softmax(predictions, dim=-1)
            mean_probs = probs.mean(dim=0)
            uncertainty = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
            return mean_probs, uncertainty
        else:
            mean = predictions.mean(dim=0)
            variance = predictions.var(dim=0)
            return mean, variance

    def _get_ensemble_state(self) -> Dict[str, Any]:
        """
        Get internal ensemble state for serialization.
        
        Returns:
            Dictionary containing all internal state needed to restore the ensemble
        """
        return {
            "model": self.model,
            "is_fitted": self.is_fitted,
            "device": self.device,
            "likelihood": self.likelihood,
            "pretrained": self.pretrained,
            "kronecker_factors": self.kronecker_factors,
            "sampling_factors": self.sampling_factors,
            "dataset_size": self.dataset_size,
            "hook_handles": self.hook_handles,
            "verbose": self.verbose,
        }

    def _set_ensemble_state(self, state: Dict[str, Any]):
        """
        Set internal ensemble state from serialized dictionary.
        
        Args:
            state: Dictionary containing ensemble state
            
        Raises:
            ValueError: If the saved likelihood is not supported
        """
        if state["likelihood"] in ["classification", "regression"]:
            self.likelihood = state["likelihood"]
        else:
            raise ValueError(f"Unsupported likelihood: {state['']}")

        self.model = state["model"]
        self.is_fitted = state["is_fitted"]
        self.device = state["device"]
        self.pretrained = state["pretrained"]
        self.kronecker_factors = state["kronecker_factors"]
        self.sampling_factors = state["sampling_factors"]
        self.dataset_size = state["dataset_size"]
        self.hook_handles = state["hook_handles"]
        self.verbose = state["verbose"]
