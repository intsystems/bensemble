# Methods

## VariationalEnsemble
**`class bensemble.methods.variational_inference.VariationalEnsemble`**

Main ensemble class for Bayesian neural networks with variational inference.

### Constructor
```python
VariationalEnsemble(
    model: nn.Module,
    likelihood: Optional[nn.Module] = None,
    learning_rate: float = 1e-3,
    prior_sigma: float = 1.0,
    auto_convert: bool = True,
    **kwargs
)
```

**Parameters:**

- `model` (nn.Module): Base neural network model
- `likelihood` (nn.Module, optional): Likelihood module. If None, uses `GaussianLikelihood()`. Default: None
- `learning_rate` (float, optional): Learning rate for optimizer. Default: 1e-3
- `prior_sigma` (float, optional): Prior standard deviation for Bayesian layers. Default: 1.0
- `auto_convert` (bool, optional): Automatically convert Linear layers to BayesianLinear. Default: True
- `**kwargs`: Additional arguments passed to parent class

**Attributes:**

- `prior_sigma` (float): Prior standard deviation
- `learning_rate` (float): Learning rate
- `likelihood` (nn.Module): Likelihood module
- `optimizer` (torch.optim.Optimizer): Optimizer instance (initialized during `fit`)
- `model` (nn.Module): Bayesian model
- `is_fitted` (bool): Training status flag

### `fit`
Trains the Bayesian ensemble model.

**Parameters:**

- `train_loader` (torch.utils.data.DataLoader): Training data loader
- `val_loader` (torch.utils.data.DataLoader, optional): Validation data loader. Default: None
- `epochs` (int, optional): Number of training epochs. Default: 100
- `kl_weight` (float, optional): Weight for KL divergence term. Default: 0.1
- `verbose` (bool, optional): Print training progress. Default: True
- `**kwargs`: Additional arguments including:
  - `device` (str/torch.device): Device to use for training

**Returns:**

- `dict`: Training history with keys: `train_loss`, `nll`, `kl`

**Raises:**

- `ValueError`: If data loaders are improperly configured

### `predict`
Makes predictions with uncertainty estimates using Monte Carlo sampling.

**Parameters:**

- `X` (torch.Tensor): Input tensor of shape `(batch_size, features)`
- `n_samples` (int, optional): Number of Monte Carlo samples. Default: 100

**Returns:**

- `Tuple[torch.Tensor, torch.Tensor]`: Mean and standard deviation predictions

**Raises:**

- `RuntimeError`: If model is not fitted

**Note:**
Predictions are moved to CPU and detached from computation graph.

### `sample_models`
Samples deterministic models from the variational posterior.

**Parameters:**

- `n_models` (int, optional): Number of models to sample. Default: 10

**Returns:**

- `List[nn.Module]`: List of sampled neural network models

**Raises:**

- `RuntimeError`: If model is not fitted

**Note:**
Each sampled model has fixed weights (sampling disabled) and is in evaluation mode.

### Notes

1. **Device Management:** The ensemble automatically moves models to the appropriate device during training and prediction.

2. **Sampling Behavior:**
    - Training: Sampling enabled with local reparameterization
    - Prediction: Sampling enabled for MC estimates
    - Sampled models: Sampling permanently disabled (deterministic)

3. **KL Weighting:** The `kl_weight` parameter controls the trade-off between data fit and model complexity.

4. **Noise Estimation:** When using `GaussianLikelihood`, the noise sigma is learned during training and used for prediction uncertainty.

5. **Memory:** `sample_models` creates deep copies of the model, which may be memory intensive for large models.

---

## LaplaceApproximation
**`class bensemble.methods.laplace_approximation.LaplaceApproximation`**

### Constructor

```python
def __init__(
    self,
    model: nn.Module,
    pretrained: bool = True,
    likelihood: str = "classification",
    verbose: bool = False,
)
```

Initializes the Laplace approximation ensemble.

**Parameters:**

- `model` (nn.Module): The neural network model to apply Laplace approximation to.
- `pretrained` (bool, optional): Whether the model is already trained. If `False`, the model will be trained during `fit()`. Default: `True`.   
- `likelihood` (str, optional): The type of likelihood function. Must be either `"classification"` or `"regression"`. Default: `"classification"`.
- `verbose` (bool, optional): Whether to print progress information during training and posterior computation. Default: `False`.

**Raises:**

- `ValueError`: If `likelihood` is not `"classification"` or `"regression"`.

**Example:**
```python
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))
la = LaplaceApproximation(model, likelihood="classification", verbose=True)
```

### `toggle_verbose`

```python
def toggle_verbose(self) -> None
```

Toggles verbose mode on or off.

**Example:**
```python
la.toggle_verbose()  # Turns verbose on if off, off if on
```

---

### `fit`

```python
def fit(
    self,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 100,
    lr: float = 1e-3,
    prior_precision: float = 1.0,
    num_samples: int = 1000,
) -> Dict[str, List[float]]
```

Trains the model (if not pretrained) and computes the Laplace approximation posterior.

**Parameters:**

- `train_loader` (DataLoader): DataLoader for training data.
- `val_loader` (DataLoader, optional): DataLoader for validation data. Currently unused but included for API consistency.
- `num_epochs` (int, optional): Number of training epochs if model is not pretrained. Default: `100`.
- `lr` (float, optional): Learning rate for training if model is not pretrained. Default: `1e-3`.
- `prior_precision` (float, optional): Precision (inverse variance) of the Gaussian prior. Default: `1.0`.
- `num_samples` (int, optional): Number of samples to use for estimating Kronecker factors. Default: `1000`.

**Returns:**

- `Dict[str, List[float]]`: Training history containing loss values.

**Example:**
```python
history = la.fit(
    train_loader=train_loader,
    num_epochs=50,
    lr=0.001,
    prior_precision=0.1,
    num_samples=500
)
```

---

### `compute_posterior`

```python
def compute_posterior(
    self,
    train_loader: DataLoader,
    prior_precision: float = 0.0,
    num_samples: int = 1000,
) -> None
```

Computes the Kronecker-factored Laplace approximation posterior.

This method estimates the posterior distribution by computing Kronecker-factored curvature matrices for each linear layer in the network.

**Parameters:**

- `train_loader` (DataLoader): DataLoader containing training data for posterior estimation.
- `prior_precision` (float, optional): Precision (inverse variance) of the Gaussian prior. Default: `0.0`.
- `num_samples` (int, optional): Number of samples to use for estimating Kronecker factors. Default: `1000`.

**Example:**
```python
la.compute_posterior(
    train_loader=train_loader,
    prior_precision=1.0,
    num_samples=2000
)
```

---

### `sample_models`

```python
def sample_models(
    self,
    n_models: int = 10,
    temperature: float = 1.0
) -> List[nn.Module]
```

Samples weight matrices from the matrix normal posterior distribution.

**Parameters:**

- `n_models` (int, optional): Number of model samples to generate. Default: `10`.
- `temperature` (float, optional): Scaling factor for the covariance during sampling. Higher values increase diversity. Default: `1.0`.

**Returns:**

- `List[nn.Module]`: List of sampled model instances with different weight configurations.

**Example:**
```python
sampled_models = la.sample_models(n_models=5, temperature=0.5)
for model in sampled_models:
    predictions = model(test_data)
```

---

### `predict`

```python
def predict(
    self,
    X: torch.Tensor,
    n_samples: int = 100,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]
```

Computes predictive distribution using Monte Carlo sampling from the posterior.

**Parameters:**

- `X` (torch.Tensor): Input tensor of shape `(batch_size, input_dim)`.
- `n_samples` (int, optional): Number of Monte Carlo samples to draw. Default: `100`.
- `temperature` (float, optional): Scaling factor for the covariance during sampling. Default: `1.0`.

**Returns:**

- `Tuple[torch.Tensor, torch.Tensor]`:
  - For classification: `(mean_probabilities, predictive_entropy)`
  - For regression: `(mean_prediction, predictive_variance)`

**Example:**
```python
# For classification
mean_probs, uncertainty = la.predict(X_test, n_samples=50)

# For regression
mean_pred, variance = la.predict(X_test, n_samples=50)
```

---

### Notes

1. **Model Requirements:** The method currently supports `nn.Linear` layers. Other layer types are ignored.

2. **Device Handling:** The class automatically detects and uses the same device as the input model.

3. **State Management:** After calling `fit()` or `compute_posterior()`, the internal state (`kronecker_factors`, `sampling_factors`, etc.) is populated and ready for sampling.

4. **Memory Usage:** Sampling large models with many Monte Carlo samples may require significant memory.

5. **Likelihood Types:**

    - `"classification"`: Uses cross-entropy loss with softmax output.
    - `"regression"`: Uses mean squared error loss.

---

## VariationalRenyi

**`class bensemble.methods.variational_renyi.VariationalRenyi`**

### Constructor

```python
def __init__(
    self,
    model: nn.Module,
    alpha: float = 1.0,
    prior_sigma: float = 1.0,
    initial_rho: float = -2.0,
    **kwargs
)
```

Initializes the VariationalRenyi model.

**Parameters:**

- `model` (nn.Module): A deterministic PyTorch neural network model. Only Linear layers will be made Bayesian.
- `alpha` (float, optional): Rényi alpha parameter. When α=1, this reduces to standard Variational Inference. Default: 1.0.
- `prior_sigma` (float, optional): Standard deviation of the Gaussian prior distribution for weights. Default: 1.0.
- `initial_rho` (float, optional): Initial value for the rho parameter used in variational weight parameterization. Default: -2.0.
- `**kwargs`: Additional arguments passed to the parent `BaseBayesianEnsemble` class.

### `fit`

```python
def fit(
    self,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 100,
    lr: float = 5e-4,
    n_samples: int = 10,
    grad_clip: Optional[float] = 5.0,
    **kwargs
) -> Dict[str, List[float]]
```

Trains the model using the Variational Rényi bound.

**Parameters:**

- `train_loader` (DataLoader): DataLoader for training data.
- `val_loader` (DataLoader, optional): DataLoader for validation data. If provided, validation loss is computed each epoch. Default: None.
- `num_epochs` (int, optional): Number of training epochs. Default: 100.
- `lr` (float, optional): Learning rate for the Adam optimizer. Default: 5e-4.
- `n_samples` (int, optional): Number of Monte Carlo samples used to approximate the loss each iteration. Default: 10.
- `grad_clip` (float, optional): Maximum absolute value for gradient clipping. If None, no clipping is applied. Default: 5.0.
- `**kwargs`: Additional keyword arguments (not currently used).

**Returns:**

- `Dict[str, List[float]]`: Training history containing:
  - `"train_loss"`: List of training losses for each epoch.
  - `"val_loss"`: List of validation losses for each epoch (if val_loader provided).

### `predict`

```python
def predict(
    self, 
    X: torch.Tensor, 
    n_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]
```

Makes predictions using the trained model with uncertainty quantification.

**Parameters:**

- `X` (torch.Tensor): Input tensor of shape (batch_size, ...).
- `n_samples` (int, optional): Number of stochastic forward passes to sample. Default: 100.

**Returns:**

- `Tuple[torch.Tensor, torch.Tensor]`: A tuple containing:
  1. Mean prediction across all samples.
  2. All sampled predictions of shape (n_samples, batch_size, ...).

**Raises:**

- `RuntimeError`: If the model hasn't been fitted (call `fit()` first).

### `sample_models`

```python
def sample_models(self, n_models: int = 10) -> List[nn.Module]
```

Samples multiple deterministic models from the learned variational distribution.

**Parameters:**

- `n_models` (int, optional): Number of models to sample. Default: 10.

**Returns:**

- `List[nn.Module]`: List of sampled PyTorch models with frozen weights.

### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Performs a single stochastic forward pass through the network.

**Parameters:**

- `x` (torch.Tensor): Input tensor.

**Returns:**

- `torch.Tensor`: Model output. For regression: shape (batch_size, 1); for classification: shape (batch_size, num_classes).

### Notes

- The model converts all `nn.Linear` layers in the provided template to Bayesian layers with learnable mean (mu) and variance (rho) parameters.
- For regression tasks, the likelihood uses a Gaussian distribution with fixed sigma=0.1.
- For classification tasks, the likelihood uses cross-entropy loss.
- When `alpha=1.0`, the method reduces to standard Variational Inference (ELBO).
- The `is_fitted` attribute is set to `True` after successful training.
- The class inherits from `BaseBayesianEnsemble` and may have additional methods from the parent class.

---

## ProbabilisticBackpropagation
**`class bensemble.methods.probabilistic_backpropagation.ProbabilisticBackpropagation`**

### Constructor
```python
def __init__(
    self,
    model: Optional[nn.Module] = None,
    layer_sizes: Optional[List[int]] = None,
    noise_alpha: float = 6.0,
    noise_beta: float = 6.0,
    weight_alpha: float = 6.0,
    weight_beta: float = 6.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
)
```

**Parameters:**

- `model` (Optional[nn.Module]): Pre-initialized PBP model. If None, `layer_sizes` must be provided.
- `layer_sizes` (Optional[List[int]]): List specifying the number of neurons in each layer (e.g., `[input_dim, hidden_dim, output_dim]`).
- `noise_alpha` (float): Prior shape parameter for observation noise Gamma distribution. Default: 6.0
- `noise_beta` (float): Prior rate parameter for observation noise Gamma distribution. Default: 6.0
- `weight_alpha` (float): Prior shape parameter for weight precision Gamma distribution. Default: 6.0
- `weight_beta` (float): Prior rate parameter for weight precision Gamma distribution. Default: 6.0
- `dtype` (torch.dtype): Data type for computations. Default: torch.float64
- `device` (Optional[torch.device]): Device for computations (CPU/GPU). Default: torch.device("cpu")

**Raises:**

- `ValueError`: If neither `model` nor `layer_sizes` are specified.

### fit
```python
def fit(
    self,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 100,
    step_clip: Optional[float] = 2.0,
    prior_refresh: int = 1,
    **kwargs
) -> Dict[str, List[float]]
```

Train the PBP model using Assumed Density Filtering (ADF).

**Parameters:**

- `train_loader` (DataLoader): DataLoader for training data.
- `val_loader` (Optional[DataLoader]): DataLoader for validation data. Default: None
- `num_epochs` (int): Number of training epochs. Default: 100
- `step_clip` (Optional[float]): Gradient clipping value for ADF updates. If None, no clipping. Default: 2.0
- `prior_refresh` (int): Number of prior refresh steps per epoch. Default: 1
- `**kwargs`: Additional arguments (ignored, for compatibility).

**Returns:**

- `Dict[str, List[float]]`: Training history containing:
  - `train_rmse`: List of training RMSE values per epoch
  - `train_nlpd`: List of training Negative Log Predictive Density (NLPD) values per epoch
  - `val_rmse`: List of validation RMSE values per epoch (if val_loader provided)
  - `val_nlpd`: List of validation NLPD values per epoch (if val_loader provided)

### predict
```python
def predict(
    self,
    X: torch.Tensor,
    n_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]
```

Generate predictive distribution for input data.

**Parameters:**

- `X` (torch.Tensor): Input tensor of shape `(n_samples, n_features)`.
- `n_samples` (int): Number of Monte Carlo samples to draw from predictive distribution. Default: 100

**Returns:**

- `Tuple[torch.Tensor, torch.Tensor]`: Tuple containing:
  - `mean`: Predictive mean tensor of shape `(n_samples, output_dim)`
  - `samples`: Monte Carlo samples tensor of shape `(n_samples, n_samples, output_dim)`

**Raises:**

- `RuntimeError`: If model has not been fitted.

### noise_variance
```python
def noise_variance(self) -> torch.Tensor
```

Get the estimated observation noise variance.

**Returns:**

- `torch.Tensor`: Scalar tensor representing the noise variance.

### sample_models
```python
def sample_models(self, n_models: int = 10) -> List[nn.Module]
```

Sample deterministic neural networks from the posterior distribution.

**Parameters:**

- `n_models` (int): Number of models to sample. Default: 10

**Returns:**

- `List[nn.Module]`: List of sampled models as `nn.Sequential` instances with ReLU activations.

### Notes

1. The implementation uses double precision (`torch.float64`) by default for numerical stability.
2. Training is performed using Assumed Density Filtering (ADF) with online updates.
3. The model provides both epistemic (model) and aleatoric (noise) uncertainty.
4. The prior refresh mechanism helps maintain stable hyperparameter updates during training.
5. The `predict` method returns both the predictive mean and Monte Carlo samples for full distribution analysis.