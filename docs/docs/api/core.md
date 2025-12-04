# Core

## BaseBayesianEnsemble
**`class bensemble.core.base.BaseBayesianEnsemble`**

Base class for all Bayesian ensemble methods.

### Constructor

```python
def __init__(self, model: nn.Module, **kwargs):
```

**Parameters**

- **model** (`nn.Module`): Base PyTorch model architecture to use for ensemble members
- **kwargs**: Additional implementation-specific parameters

**Attributes**

- **model** (`nn.Module`): Reference to the base model architecture
- **is_fitted** (`bool`): Flag indicating whether the ensemble has been trained
- **ensemble** (`list`): Container for ensemble members (implementation-specific)

### `fit`

```python
@abc.abstractmethod
def fit(
    self,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    **kwargs,
) -> Dict[str, List[float]]:
```

Trains the ensemble using the provided data loaders.

**Parameters**

- **train_loader** (`DataLoader`): DataLoader for training data
- **val_loader** (`DataLoader`, optional): DataLoader for validation data
- **kwargs**: Additional training parameters (implementation-specific)

**Returns**

- `Dict[str, List[float]]`: Training history/metrics (implementation-specific format)

**Notes**

- Must be implemented by all subclasses
- Should update `self.is_fitted` to `True` upon successful training

### `predict`

```python
@abc.abstractmethod
def predict(
    self, X: torch.Tensor, n_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
```

Makes predictions with uncertainty estimates.

**Parameters**

- **X** (`torch.Tensor`): Input tensor of shape `(batch_size, ...)`
- **n_samples** (`int`, default=100): Number of forward passes/samples for uncertainty estimation

**Returns**

- `Tuple[torch.Tensor, torch.Tensor]`: 
  - First tensor: Predictions (typically mean)
  - Second tensor: Uncertainty estimates (e.g., variance, standard deviation)

**Notes**

- Should raise an error if `self.is_fitted` is `False`
- The exact form of uncertainty estimates depends on the implementation

### `sample_models`

```python
@abc.abstractmethod
def sample_models(self, n_models: int = 10) -> List[nn.Module]:
```

Samples individual models from the posterior distribution.

**Parameters**

- **n_models** (`int`, default=10): Number of models to sample

**Returns**

- `List[nn.Module]`: List of sampled PyTorch models

**Notes**

- Intended for online generation and maintenance of ensemble members
- Sampled models should be ready for inference

### `_get_ensemble_state`

```python
@abc.abstractmethod
def _get_ensemble_state(self) -> Dict[str, Any]:
```

Gets the internal state of the ensemble for serialization.

**Returns**

- `Dict[str, Any]`: Dictionary containing all necessary state information

**Notes**

- Used internally by `save()` method
- Implementation should include all parameters needed to restore the ensemble

### `_set_ensemble_state`

```python
@abc.abstractmethod
def _set_ensemble_state(self, state: Dict[str, Any]):
```

Restores the internal state of the ensemble from serialized data.

**Parameters**

- **state** (`Dict[str, Any]`): Dictionary containing saved state information

**Notes**

- Used internally by `load()` method
- Should handle version compatibility if needed

### `save`

```python
def save(self, path: str):
```

Saves the trained ensemble to disk.

**Parameters**

- **path** (`str`): File path where the ensemble will be saved

**Saves**

- Base model state dictionary
- Ensemble state (via `_get_ensemble_state()`)
- `is_fitted` flag

**File Format**

- PyTorch checkpoint file (`.pt` or `.pth`)

### `load`

```python
def load(self, path: str):
```

Loads a trained ensemble from disk.

**Parameters**

- **path** (`str`): File path to the saved ensemble

**Loads**

- Base model state dictionary
- Ensemble state (via `_set_ensemble_state()`)
- `is_fitted` flag

**Raises**

- `FileNotFoundError`: If the specified path doesn't exist
- Runtime errors if the saved format is incompatible

#### Implementation Notes

1. **Subclassing**: All abstract methods must be implemented by subclasses
2. **Device Management**: Implementations should handle device placement (CPU/GPU)
3. **State Management**: Ensure `_get_ensemble_state()` and `_set_ensemble_state()` are comprehensive
4. **Error Handling**: Check `is_fitted` flag in `predict()` and `sample_models()`
5. **Serialization**: Consider versioning for saved models to handle future changes