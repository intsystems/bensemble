# Probabilistic Backpropagation

The implementation of PBP (Hernández-Lobato & Adams, 2015) uses Assumed Density Filtering (ADF) in an online fashion.

Means and variances are propagated analytically through the network. For ReLU activations, we use exact moment-matching functions relying on the PDF/CDF of the standard normal distribution.

Weights are updated by matching the moments of the tilted distribution $q(w)p(y|x,w)$. We compute gradients of the log-partition function $\log Z$ to update $\mu$ and $\Sigma$ directly, completely bypassing standard Stochastic Gradient Descent (SGD).

## Usage

`PBPEngine` owns the training loop, so there is no optimizer to configure. It works in `float64`: `fit` and `forward_moments` convert their inputs, but the networks returned by `build_ensemble` do not, so keep the data in `float64` throughout.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from bensemble.methods.probabilistic_backpropagation import PBPEngine

x = torch.randn(200, 4, dtype=torch.float64)
y = x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(200, 1, dtype=torch.float64)
loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

engine = PBPEngine(layer_sizes=[4, 32, 1])
history = engine.fit(loader, num_epochs=5)
print(history["train_rmse"])
```

Predictions come in two forms. The analytic one propagates moments through the network and returns a mean and a variance per input:

```python
mean, var = engine.model.forward_moments(x[:5])
```

Or sample explicit networks from the fitted posterior and treat them like any other ensemble:

```python
ensemble = engine.build_ensemble(n_members=10)
member_preds = ensemble.predict_members(x[:5])  # [10, 5, 1]
```

The engine builds a `PBPNet` from `layer_sizes`, a stack of `ProbLinear` layers. Unlike an ordinary linear layer, each one holds a posterior mean `m` and variance `v` for every weight, and that is what `fit` updates:

```python
from bensemble.methods.probabilistic_backpropagation import ProbLinear

for layer in engine.model.layers:
    assert isinstance(layer, ProbLinear)
    print(tuple(layer.m.shape), tuple(layer.v.shape))
# (32, 5) (32, 5)
# (1, 33) (1, 33)
```

To control the architecture yourself, build the network first and hand it over:

```python
from bensemble.methods.probabilistic_backpropagation import PBPNet

net = PBPNet(layer_sizes=[4, 32, 32, 1])
engine = PBPEngine(model=net)
```

---

José Miguel Hernández-Lobato, Ryan Adams [*"Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks"*](https://proceedings.mlr.press/v37/hernandez-lobatoc15.html) (2015)
