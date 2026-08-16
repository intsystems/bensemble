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

Easily ensemble standard PyTorch models, calibrate them post-hoc using Temperature Scaling, and decompose their uncertainty to detect Out-Of-Distribution (OOD) data.

```python
import torch
import torch.nn as nn
from bensemble.core.ensemble import Ensemble
from bensemble.calibration.scaling import TemperatureScaling
from bensemble.uncertainty import decompose_classification_uncertainty
from bensemble.metrics import expected_calibration_error

# 1. Create a Deep Ensemble from standard trained PyTorch models
models = [nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3)) for _ in range(5)]
ensemble = Ensemble.from_models(models)

# 2. Calibrate the ensemble using a hold-out validation set
val_logits, val_labels = torch.randn(100, 3), torch.randint(0, 3, (100,))
scaler = TemperatureScaling(init_temp=1.5).fit(val_logits, val_labels)

# 3. Predict on test data
test_x = torch.randn(10, 10)
# Returns shape: [5 models, 10 batch_size, 3 classes]
logits = scaler(ensemble.predict_members(test_x)) 
probs = torch.softmax(logits, dim=-1)

# 4. Decompose Uncertainty & Evaluate
total, aleatoric, epistemic = decompose_classification_uncertainty(probs)
ece = expected_calibration_error(probs.mean(dim=0), val_labels[:10])

print(f"Calibration Error (ECE): {ece:.4f}")
print(f"Epistemic Uncertainty (OOD awareness): {epistemic.mean().item():.4f}")
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
