# Additional Classes

## BayesianLinear
**`class bensemble.methods.variational_inference.BayesianLinear`**

Bayesian linear layer with mean-field variational inference.

### Constructor
```python
BayesianLinear(in_features, out_features, prior_sigma=1.0)
```

**Parameters:**

- `in_features` (int): Size of each input sample
- `out_features` (int): Size of each output sample
- `prior_sigma` (float, optional): Standard deviation of the Gaussian prior. Default: 1.0

**Attributes:**

- `in_features` (int): Input feature dimension
- `out_features` (int): Output feature dimension
- `prior_sigma` (float): Prior standard deviation
- `sampling` (bool): Sampling mode flag (True for training/MC, False for mean prediction)
- `w_mu` (torch.Tensor): Mean of weight variational distribution
- `w_rho` (torch.Tensor): Raw parameter for weight standard deviation
- `b_mu` (torch.Tensor): Mean of bias variational distribution
- `b_rho` (torch.Tensor): Raw parameter for bias standard deviation

### `forward`
Performs forward pass through the Bayesian linear layer.

**Parameters:**

- `x` (torch.Tensor): Input tensor of shape `(batch_size, in_features)`

**Returns:**

- `torch.Tensor`: Output tensor of shape `(batch_size, out_features)`

**Behavior:**

- In sampling mode (`sampling=True`): Uses local reparameterization trick
- In mean mode (`sampling=False`): Uses only the mean parameters

### `kl_divergence`
Computes KL divergence between variational posterior and prior.

**Returns:**

- `torch.Tensor`: KL divergence scalar value

**Formula:**
KL(q(w|θ) || p(w)) where p(w) = N(0, prior_sigma²) and q(w|θ) = N(μ, σ²)

---

## GaussianLikelihood
**`class bensemble.methods.variational_inference.GaussianLikelihood`**

Gaussian likelihood module for regression tasks with learnable noise parameter.

### Constructor
```python
GaussianLikelihood(init_log_sigma=-2.0)
```

**Parameters:**

- `init_log_sigma` (float, optional): Initial value for log(sigma). Default: -2.0

**Attributes:**

- `log_sigma` (torch.nn.Parameter): Learnable log standard deviation parameter

### `forward`
Computes negative log-likelihood for Gaussian distribution.

**Parameters:**

- `preds` (torch.Tensor): Model predictions
- `target` (torch.Tensor): Ground truth targets

**Returns:**

- `torch.Tensor`: Negative log-likelihood scalar value

**Formula:**
-0.5 * (target - preds)² / σ² + log(σ)

### `get_noise_sigma`
Gets the current noise standard deviation value.

**Returns:**
- `float`: Noise sigma value (detached from computation graph)

---

## PBPNet
**`class bensemble.methods.probabilistic_backpropagation.PBPNet`**

A neural network architecture based on `ProbLinear` layers with analytical moment propagation for Probabilistic Backpropagation (PBP).

This network implements moment propagation through Gaussian approximations, enabling efficient Bayesian inference without sampling during forward passes. It uses analytical formulas to propagate means and variances through linear transformations and ReLU activations.

### Constructor
```python
def __init__(
    self,
    layer_sizes: List[int],
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
)
```

Initialize a PBPNet with specified architecture.

**Parameters:**

- `layer_sizes` (List[int]): List specifying the number of neurons in each layer.
  - Example: `[input_dim, hidden_dim1, hidden_dim2, ..., output_dim]`
  - Minimum length: 2 (input and output layers)
- `dtype` (torch.dtype): Data type for network parameters and computations. Default: `torch.float64`
- `device` (Optional[torch.device]): Device for network parameters and computations (CPU/GPU). Default: CPU device

**Raises:**

- `ValueError`: If `layer_sizes` has less than 2 elements.

**Example:**
```python
# Create a network with 10 inputs, 2 hidden layers (50 and 20 neurons), and 1 output
model = PBPNet(layer_sizes=[10, 50, 20, 1], dtype=torch.float64, device=torch.device("cuda"))
```

### `forward_moments`
```python
def forward_moments(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.torch.Tensor]
```

Propagate input through the network while tracking means and variances analytically.

This method implements the core moment propagation algorithm for PBP:

1. Appends a bias term (constant 1) to the input with zero variance
2. For each layer:
   - Computes mean and variance of linear transformation
   - For non-final layers: applies ReLU activation with moment matching
   - For final layer: returns linear transformation moments

**Parameters:**

- `x` (torch.Tensor): Input tensor of shape `(batch_size, input_dim)`.
  - Must have 2 dimensions (batch dimension required)
  - Batch size must be ≥ 1

**Returns:**

- `Tuple[torch.Tensor, torch.Tensor]`: Tuple containing:
  - `mean`: Output mean tensor of shape `(batch_size, output_dim)`
  - `variance`: Output variance tensor of shape `(batch_size, output_dim)`

**Raises:**

- `AssertionError`: If input tensor does not have 2 dimensions or batch size is 0.

**Mathematical Details:**
For a linear layer with input moments `(m_z, v_z)` and layer parameters `(m, v)`:
- Output mean: `m_a = (m_z @ m^T) / sqrt(d)` where `d = input_dim + 1`
- Output variance: `v_a = (v_z @ m^2 + m_z^2 @ v + v_z @ v) / d`

For ReLU activation (non-final layers):
- Uses `relu_moments()` function to compute post-activation moments
- Bias term is added back after activation

**Example:**
```python
# Create network
model = PBPNet(layer_sizes=[5, 10, 1])

# Input batch of 3 samples, 5 features each
x = torch.randn(3, 5, dtype=torch.float64)

# Get predictive moments
mean, variance = model.forward_moments(x)
# mean.shape: (3, 1)
# variance.shape: (3, 1)
```

### Usage Notes

1. **Precision**: Defaults to `torch.float64` for numerical stability in moment computations.
2. **Device Management**: Automatically moves input tensors to the network's device and dtype.
3. **Batch Processing**: Efficiently processes batches using matrix operations.
4. **No Training Interface**: This class only provides forward moment propagation; training is handled by `ProbabilisticBackpropagation`.