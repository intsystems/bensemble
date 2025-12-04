# Probabilistic Backpropagation

*Pro tip: check out our [probabilistic backpropagation demo](https://github.com/intsystems/bensemble/blob/master/notebooks/pbp_probabilistic_backpropagation_test.ipynb).*

We implement the probabilistic backpropagation (or Bayes by Backprop) method described by [Jose Miguel Hernandez-Lobato and Ryan Adams (2015)](https://proceedings.mlr.press/v37/hernandez-lobatoc15.html) in the [`bensemble.methods.probabilistic_backpropagation.ProbabilisticBackpropagation`](../../api/methods/#probabilisticbackpropagation) class (phew, that's a long one). This is a method that combines the backpropagation algorithm with Bayesian machine learning for efficient posterior optimization during model training. 

*Note: currently the algorithm is implemented for regression with a Gaussian likelihood.*

## How it works

Probabilistic Backpropagation (PBP) pushes the classical “uncertainty in weights” idea quite literally. PBP maintains a Gaussian distribution for model weights:
$$
w_{ij} = \mathcal{N}(\mu_{ij}, \sigma^2_{ij}).
$$
Noise and weight precisions are also treated as random with gamma posterior distributions. Conceptually, for each input the network does not produce a single scalar, but an approximate predictive distribution:
$$
p(y | \mathbf{x}, q) \approx \mathcal{N}(y | \mu(\mathbf{x}), \sigma^2(\mathbf{x})),
$$
where the distribution parameters are computed via propagation of means and variances through the network layers.

Learning then proceeds in an assumed density filtering style. For each data point, we take the current approximate posterior, multiply it by the likelihood of that point, and project the result back to the Gaussian family:
$$
q_\text{new}(\boldsymbol{\theta}) \propto q_\text{old}(\theta)p(y | \mathbf{x}, \boldsymbol{\theta}) \quad \text{(projected back to Gaussians)}.
$$

## Usage

### Initialization

First of all, you'll need to create a `ProbabilisticBackpropagation` instance. This is done slightly differently since at its core, the `ProbabilisticBackpropagation` has a [`bensemble.methods.probabilistic_backpropagation.PBPNet`](../../api/methods/#PBPNet) instance which is a network of probabilistic linear layers with ReLU activations. The simplest way to use the method is to specify layer sizes directly in the `layer_sizes` parameter:
```python
from bensemble.methods.probabilistic_backpropagation import ProbabilisticBackpropagation

# Create ProbabilisticBackpropagation instance
ensemble = ProbabilisticBackpropagation(layer_sizes=[1, 16, 16, 1])
```
Alternatively, you can create a `PBPNet` model and then specify it in the `model` parameter of the `ProbabilisticBackpropagation` constructor (see the [API section](../../api/methods/#PBPNet) for more info on PBPNet).

You can also specify several optional parameters:

- `noise_alpha: float = 6.0` - the alpha parameter of the Gamma distribution over the weight noise.
- `noise_beta: float = 6.0`  - the beta parameter of the Gamma distribution over the weight noise.
- `weight_alpha: float = 6.0` - the alpha parameter of the Gamma distribution over the weight precision.
- `weight_beta: float = 6.0` - the beta parameter of the Gamma distribution over the weight precision.
- `dtype: torch.dtype = torch.float64` - value type for network.
- `device: Optional[torch.device] = None` - model device.

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
- `step_clip: Optional[float] = 2.0` - value for step-wise clipping during training.

The `fit` method returns a `Dict[str, List[float]]` containing training and validation RMSE and NLPD values acquired during model training:
```python
training_history.keys()
> ['train_rmse', 'train_nlpd', 'val_rmse', 'val_nlpd']
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

*For more information on class methods and parameters, visit the [API section](../../api/methods/#probabilisticbackpropagation).*