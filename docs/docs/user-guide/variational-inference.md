# Practical Variational Inference

*Pro tip: check out our [practical variational inference demo](https://github.com/intsystems/bensemble/blob/master/notebooks/pvi_demo.ipynb).*

We implement the practical variational inference method described by [Graves (2011)](https://papers.nips.cc/paper_files/paper/2011/hash/7eb3c8be3d411e8ebfab08eba5f49632-Abstract.html) in the [`bensemble.methods.variational_inference.VariationalEnsemble`](../../api/methods/#variationalensemble) class. This is a simple method of in-training posterior approximation for linear neural networks that uses independent Gaussian distributions over the weights of the network.

## How it works

The **practical variational inference** (PVI) methods follows the "Bayesian layers" approach: instead of treating the whole network as a single variational object, it replaces ordinary linear layers by Bayesian linear layers and adds a simple Gaussian likelihood on top. Each weight in a linear layer is assumed to have the following distribution:
$$
w_{ij} = \mathcal{N}(\mu_{ij}, \sigma^2_{ij})
$$
with a fixed Gaussian prior. We also implement the **local reparametrization trick** to allow for standard backpropagation through the Bayesian network and effeicient sampling: for an input mini-batch, the pre-activation means and variances are 
$$
\boldsymbol{\gamma} = \mathbf{xW}\boldsymbol{\mu}^\top, \quad \boldsymbol{\delta} = \mathbf{xW}\boldsymbol{\sigma}^{2\top} + \text{bias term},
$$
and activations are drawn as
$$
\mathbf{z} = \boldsymbol{\gamma} + \boldsymbol{\varepsilon}\odot\boldsymbol{\delta}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, 1).
$$
The model is then optimized via a modified loss function that incorporates KL-divergence.

## Usage

### Initialization

First of all, you'll need to create a `VariationalEnsemble` instance. This can be done as simple as:
```python
from bensemble.methods.variational_inference import VariationalEnsemble

# Create your nn.Module model
model = ...

# Create VariationalEnsemble instance
ensemble = VariationalEnsemble(model)
```

You can also specify several optional parameters:

- `likelihood: Optional[nn.Module] = None` - custom likelihood function for the model. By default a Gaussian likelihood based on MSE for regression is used (see [`GaussianLikelihood`](../../api/methods/#gaussianlikelihood) in the API section for details).
- `learning_rate: float = 1e-3` - learning rate for model training.
- `prior_sigma: float = 1.0` - the variance of the prior distribution, which in this model is assumed to be a zero-mean Gaussian distribution.

### Training

For model training, it is sufficient to specify just a training data `DataLoader`:
```python
from torch.utils.data import DataLoader

# Create train dataset
train_data = ...

# Create DataLoader
train_loader = DataLoader(data, ...)

# Train ensemble
ensemble.fit(train_loader)
```

You can also add several optional parameters, including:

- `epochs: int = 100`: the number of training epochs.
- `kl_weight: float = 0.1`: the weight of the KL-divergence in the loss function.
- `verbose: bool = True`: option to turn training verbose on or off.

### Prediction and model sampling

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

*For more information on class methods and parameters, visit the [API section](../../api/methods/#variationalensemble).*