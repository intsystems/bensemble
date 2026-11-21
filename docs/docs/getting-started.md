# Getting Started

To get started with Bensemble, all you'll need is Python 3.10+ and PyTorch.

---

## Installation

You can install `bensemble` using pip:

```bash
pip install bensemble
```

Or, using [uv](https://github.com/astral-sh/uv) for lightning-fast installation:

```bash
uv pip install bensemble
```

---

## Core Workflows

Bensemble is designed to be flexible. You can either combine and calibrate your existing standard PyTorch models or build Bayesian neural networks from scratch.

### Workflow 1: Ensembling, Calibration & Uncertainty

Train three classifiers on synthetic data. Fit a temperature to the ensemble's probabilities on the validation split, then compare calibration on the test split.

```python
import torch
from torch import nn
from bensemble.calibration.scaling import TemperatureScaling
from bensemble.core.ensemble import Ensemble
from bensemble.metrics import expected_calibration_error
from bensemble.uncertainty import decompose_classification_uncertainty

torch.manual_seed(42)
# Two classes with a noisy linear decision boundary.
x = torch.randn(768, 2)
y = (x.sum(dim=1) + 0.5 * torch.randn(len(x)) > 0).long()
x_train, x_val, x_test = x.split(256)
y_train, y_val, y_test = y.split(256)

models = [nn.Linear(2, 2) for _ in range(3)]
for model in models:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    for _ in range(20):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x_train), y_train).backward()
        optimizer.step()
ensemble = Ensemble.from_models(models).eval()

# Average the members' probabilities. Their logarithm gives logits for the mixture.
with torch.no_grad():
    val_probs = ensemble.predict_members(x_val).softmax(dim=-1).mean(dim=0)
scaler = TemperatureScaling().fit(val_probs.log(), y_val)

with torch.no_grad():
    test_member_probs = ensemble.predict_members(x_test).softmax(dim=-1)
    test_probs = test_member_probs.mean(dim=0)
    calibrated_probs = scaler(test_probs.log()).softmax(dim=-1)

print(f"Test ECE before: {expected_calibration_error(test_probs, y_test):.3f}")
print(f"Test ECE after:  {expected_calibration_error(calibrated_probs, y_test):.3f}")
_, _, epistemic = decompose_classification_uncertainty(test_member_probs)
print(f"Mean epistemic uncertainty: {epistemic.mean().item():.4f}")
```

---

### Workflow 2: Variational Inference

Build a Bayesian Neural Network from scratch using our custom layers with the Local Reparameterization Trick and optimize the variational objective (ELBO).

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
criterion = VariationalLoss(likelihood, alpha=1.0)  # ELBO

optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)

# 3. Standard PyTorch Training Loop
model.train()
for epoch in range(50):  # Dummy loop
    x, y = torch.randn(10, 10), torch.randn(10, 1)
    optimizer.zero_grad()
    loss = criterion(model(x), y, get_total_kl(model))
    loss.backward()
    optimizer.step()

# 4. Predict with Uncertainty
mean, std = predict_with_uncertainty(model, torch.randn(5, 10), num_samples=100)
print(f"Prediction: {mean[0].item():.2f} ± {std[0].item():.2f}")
```
