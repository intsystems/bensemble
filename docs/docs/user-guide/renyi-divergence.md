# Renyi Divergence Variational Inference

*Pro tip: check out our [Renyi divergence variational inference demo](https://github.com/intsystems/bensemble/blob/master/notebooks/variatinal_renyi_demo.ipynb).*

We implement the Renyi divergence variational inference method described by [Yingzhen Li and Richard E. Turner (2016)](https://arxiv.org/abs/1602.02311) in the [`bensemble.methods.variational_renyi.VariationalRenyi`](../../api/methods/#variationalrenyi) class. This is a modification of the standard variational inference method that uses Renyi divergence instead of the usual KL-divergence for optimization.

## How it works

**Variational Rényi inference** keeps the spirit of standard variational inference method but replaces the usual KL-divergence by a whole family of divergences indexed by a parameter α. **Rényi's α-divergence** for two distributions is defined as
$$
D_\alpha(p\|q) = \frac{1}{\alpha - 1}\log\int p(\boldsymbol{\theta})^\alpha q(\boldsymbol{\theta})^{1-\alpha}d\boldsymbol{\theta}.
$$

Qualitatively, this gives a knob that controls how aggressive or conservative the variational approximation is. Once trained, sampling networks is as simple as drawing from the Gaussian and plugging the sampled weights into the base model.

## Usage

### Initialization

First of all, you'll need to create a `VariationalRenyi` instance. This can be done just by wrapping a model with it:
```python
from bensemble.methods.variational_renyi import VariationalRenyi

# Create your nn.Module model
model = ...

# Create VariationalRenyi instance
ensemble = VariationalRenyi(model)
```

You can also specify several optional parameters:

- `alpha: float = 1.0` - the α parameter of Renyi divergence.
- `prior_sigma: float = 1.0` - the variance of the prior distribution, which in this model is assumed to be a zero-mean Gaussian distribution.
- `initial_rho: float = -2.0` - the initial rho parameter of the variational weight distribution.

### Training

For model training, it is sufficient to specify just a training data `DataLoader`:
```python
from torch.utils.data import DataLoader

# Create train dataset
train_data = ...

# Create DataLoader
train_loader = DataLoader(data, ...)

# Train ensemble
training_history = ensemble.fit(train_loader)
```

You can also add several optional parameters, including:

- `val_loader: Optional[torch.utils.data.DataLoader] = None` - a DataLoader for model validation during training.
- `num_epochs: int = 100` - the number of training epochs.
- `lr: float = 5e-4` - the learning rate for training.
- `n_samples: int = 10` - the number of weights samples to use for predictions during validation.
- `grad_clip: Optional[float] = 5.0` - gradient clipping parameter.

The `fit` method returns a `Dict[str, List[float]]` containing training and validation loss values acquired during model training:
```python
# Train loss
train_loss = training_history["train_loss"]

# Validation loss
val_loss = training_history["val_loss"]
```

### Testing and model sampling

When the ensemble is trained, you can make predictions and sample model just like with any other method in Bensemble:
```python
# Create test dataset
test_data = ...

# Make predictions
for X in test_data:
    prediction, uncertainty = ensemble.predict(X, n_samples=50)
    print(prediction, uncertainty)

# Sample models
models = ensemble.sample_models(5)
```

*For more information on class methods and parameters, visit the [API section](../../api/methods/#variationalrenyi).*