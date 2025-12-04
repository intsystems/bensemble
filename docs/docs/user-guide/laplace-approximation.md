# Kronecker-Factored Laplace Approximation

*Pro tip: check out our [Kronecker-factored Laplace approximation demo](https://github.com/intsystems/bensemble/blob/master/notebooks/laplace_demo.ipynb).*

We implement the Kronecker-factored Laplace approximation method described by [Hippolyt Ritter, Aleksandar Botev and David Barber (2018)](https://discovery.ucl.ac.uk/id/eprint/10080902/1/kflaplace.pdf) in the [`bensemble.methods.laplace_approximation.LaplaceApproximation`](../../api/methods/#laplaceapproximation) class. This is a post-hoc method for posterior approximation based on the Laplace approximation of the negative log-likelihood and Kronecker factorization of the Hessian in this approximation.

## How it works

The **Laplace approximation** is based on second-order approximation of the negative log-posterior using Taylor decomposition around the MAP-estimate, which gives a Gaussian distribution centered a the MAP-estimate with the Hessian of the negative log-posterior as the covariance matrix:
$$
p(\theta | \mathcal{D}) \approx \mathcal{N}(\theta; \theta^* , \bar{H}^{-1}).
$$

Since direct Hessian computation is still quite impractical, the Hessian block for each layer is further approximated as the *Kronecker product* of the covariance of inputs and the pre-activation Hessian:
$$
H_{\lambda} \approx \mathbb{E}[\mathcal{Q}_{\lambda}] \otimes \mathbb{E}[\mathcal{H}_{\lambda}].
$$

Then we can sample weights from the posterior as from the following matrix normal distribution:
$$
W_{\lambda} \sim \mathcal{MN}(W_{\lambda}^*, \bar{\mathcal{Q}}_{\lambda}^{-1}, \bar{\mathcal{H}}_{\lambda}^{-1}).
$$
The power of this method is in its *post hoc* nature: you can plug a pretrained model into it and quickly compute the posterior approximation without having to change the training procedure. 

## Usage

### Initialization

First of all, you'll need to create a `LaplaceApproximation` instance. This can be done as simple as:
```python
from bensemble.methods.laplace_approximation import LaplaceApproximation

# Create your nn.Module model
model = ...

# Train your model
...

# Create LaplaceApproximation instance
ensemble = LaplaceApproximation(model)
```

You can also specify several optional parameters:

- `likelihood: str = "classification"` - the likelihood to use. The current implementation supports cross entropy (`"classification"`) and MSE loss (`"regression"`).
- `pretrained: bool = True` - whether the model is pretrained. If not, `.fit()` will train the model before posterior computation (see below).
- `verbose: bool = False` - whether verbose for all methods is on. Verbose can be also toggled using the `.toggle_verbose()` method.

### Posterior computation and training

Since Laplace approximation is post hoc, the default usage scenario is posterior computation over a pretrained model. You can do this by calling the `compute_posterior` method which takes a `DataLoader` as input. By default, the `fit` method does the same thing (it just calls `compute_posterior`):
```python
from torch.utils.data import DataLoader

# Create train dataset
train_data = ...

# Create DataLoader
train_loader = DataLoader(data, ...)

# Train ensemble
ensemble.compute_posterior(train_loader) # The same as: ensemble.fit(train_loader)
```

You can also add several optional parameters to `compute_posterior`, including:

- `prior_precision: float = 0.0` - the precision of the prior distribution for reqularization.
- `num_samples: int = 1000` - number of samples to use for posterior computation.

Alternatively, you can pass `pretrained=False` to the ensemble constructor and then call the `fit` method to train the model and then compute the posterior:
```python
from bensemble.methods.laplace_approximation import LaplaceApproximation
from torch.utils.data import DataLoader

# Create your nn.Module model
model = ...

# Create untrained LaplaceApproximation instance
ensemble = LaplaceApproximation(model, pretrained=False)

# Create train dataset
train_data = ...

# Create DataLoader
train_loader = DataLoader(data, ...)

# Train ensemble
ensemble.fit(train_loader)
```
The optional parameters of `fit` are:

- `num_epochs: int = 100` - the number of training epochs.
- `lr: float = 1e-3` - the learning rate for model training.
- `prior_precision: float = 1.0` - the precision of the prior distribution for reqularization.
- `num_samples: int = 1000` - number of samples to use for posterior computation.

The `fit` method returns a `Dict[str, List[float]]` containing training loss values acquired during model training:
```python
# Train loss
train_loss = training_history["train_loss"]
```

### Prediction and model sampling

When the ensemble is trained, you can make predictions and sample model just like with any other method in Bensemble. Note that model sampling takes some time in this method, so a better approach to testing may be to sample an ensemble of models and then make all the predictions with them. 
```python
# Create test dataset
test_data = ...

# Make predictions
for X in test_data:
    prediction, uncertainty = ensemble.predict(X, n_samples=20)
    print(prediction, uncertainty)

# Sample models
models = ensemble.sample_models(10)
```
The `predict` and `sample_models` also have a float `temperature` parameter which can be specified to scale the posterior distribution. 

*For more information on class methods and parameters, visit the [API section](../../api/methods/#laplaceapproximation).*