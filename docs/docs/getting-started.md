# Getting Started

To get started with Bensemble, all you'll need is Python with at least version 3.10.

## Installation

```bash
pip install bensemble
```

## Basic usage

All Bensemble classes are built as wrappers around PyTorch models that are subclasses of the `torch.nn.Module` class. All you need then is a model, a `torch.utils.data.DataLoader` for training and you're good to go!

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bensemble.layers import BayesianLinear
from bensemble.losses import VariationalLoss, GaussianLikelihood
from bensemble.utils import get_total_kl, predict_with_uncertainty

# 1. Define Model using Bayesian Layers
model = nn.Sequential(
    BayesianLinear(10, 50, prior_sigma=1.0),
    nn.ReLU(),
    BayesianLinear(50, 1, prior_sigma=1.0),
)

# 2. Define Objectives (Likelihood + Divergence)
likelihood = GaussianLikelihood()
criterion = VariationalLoss(likelihood, alpha=1.0)

optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)

# 3. Standard PyTorch Training Loop
model.train()
for epoch in range(50): # Dummy loop
    x, y = torch.randn(10, 10), torch.randn(10, 1)
    optimizer.zero_grad()
    loss = criterion(model(x), y, get_total_kl(model))
    loss.backward()
    optimizer.step()

# 4. Predict with Uncertainty
mean, std = predict_with_uncertainty(model, torch.randn(5, 10), num_samples=100)
print(f"Prediction: {mean[0].item():.2f} ± {std[0].item():.2f}")
```
