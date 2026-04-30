# Getting Started

To get started with Bensemble, all you'll need is Python 3.10+ and PyTorch.

## Installation

```bash
pip install bensemble
```

## Core Workflows

Bensemble is designed to be flexible. You can either build Bayesian models from scratch or ensemble your existing standard models.

### Workflow 1: The Bayesian Way

If you want to train a Bayesian Neural Network, simply replace standard `nn.Linear` with `BayesianLinear` and use our Variational objectives.

```python
import torch
import torch.nn as nn
from bensemble.layers import BayesianLinear
from bensemble.losses import VariationalLoss, GaussianLikelihood
from bensemble.utils import get_total_kl, predict_with_uncertainty

# 1. Define Model using Bayesian Layers
model = nn.Sequential(
    BayesianLinear(10, 50, prior_sigma=1.0),
    nn.ReLU(),
    BayesianLinear(50, 1, prior_sigma=1.0),
)

# 2. Define Objectives
likelihood = GaussianLikelihood()
criterion = VariationalLoss(likelihood, alpha=1.0) # ELBO

# 3. Standard PyTorch Training Loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(50):
    x, y = torch.randn(10, 10), torch.randn(10, 1)
    optimizer.zero_grad()
    loss = criterion(model(x), y, get_total_kl(model))
    loss.backward()
    optimizer.step()

# 4. Predict with Uncertainty
mean, std = predict_with_uncertainty(model, x_test, num_samples=100)
```

### Workflow 2: The Ensemble Way

Already have a few trained models? Bensemble makes it trivial to combine them, calibrate them, and extract uncertainty metrics.

```python
from bensemble.core.ensemble import Ensemble
from bensemble.uncertainty import decompose_classification_uncertainty

# 1. Wrap your standard trained models into an Ensemble
# models = [model1, model2, model3, ...]
ensemble = Ensemble.from_models(models)

# 2. Get standardized member predictions [M, Batch, Classes]
member_preds = ensemble.predict_members(x_test)
probs = torch.softmax(member_preds, dim=-1)

# 3. Decompose Uncertainty
total, aleatoric, epistemic = decompose_classification_uncertainty(probs)
print(f"Epistemic uncertainty: {epistemic.mean()}")
```
