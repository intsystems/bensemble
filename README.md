# 🚧 Under Active Refactoring 🚧

The library is currently undergoing a major rewrite to become more modular and PyTorch-native. The API is stabilizing!

---

# Bensemble: Modular Bayesian Deep Learning & Ensembling

[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](tests/)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://intsystems.github.io/bensemble/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://img.shields.io/badge/uv-managed-a6c489)](https://github.com/astral-sh/uv)

**Bensemble** is a production-ready, lightweight library for Bayesian Deep Learning and Neural Network Ensembling.

---

## Key Resources

| Resource | Description |
| :--- | :--- |
| 📘 **[Documentation](https://intsystems.github.io/bensemble/)** | Full API reference and user guides. |
| 📝 **[Tech Report](https://github.com/intsystems/bensemble/blob/master/paper/techreport_draft/bensemble_tech_report.pdf)** | In-depth technical details and theoretical background. |
| ✍️ **[Blog Post](https://mikhmed-nabiev.github.io/bayesian-multimodelling.html)** | Summary of the project and motivation. |
| 📊 **[Benchmarks](https://github.com/intsystems/bensemble/blob/master/notebooks/benchmark.ipynb)** | Comparison of methods on standard datasets. |

---

## Features

- **PyTorch-Native**: No hidden training loops. Use standard PyTorch to train your models, and use Bensemble for inference, ensembling, and analytics.
- **Unified Ensembling API**: Seamlessly combine explicit models (Deep Ensembles, NAS) and implicit methods (MC Dropout) via a single `Ensemble` interface.
- **Neural Ensemble Search (NES)**: State-of-the-art algorithms to automatically search for diverse architectures using NNI and Stein Variational Gradient Descent (SVGD).
- **Uncertainty Analytics**: Principled decomposition of predictive uncertainty into *aleatoric* (data noise) and *epistemic* (model ignorance) components.
- **Model Calibration & Metrics**: Evaluate models using Expected Calibration Error (ECE), Brier Score, and NLL. Fix overconfident networks post-hoc with Temperature and Vector Scaling.

---

## Installation

You can install `bensemble` using pip:

```bash
pip install bensemble
```

Or, using [uv](https://github.com/astral-sh/uv) for lightning-fast installation (recommended):

```bash
uv pip install bensemble
```

---

## Quick Start

### Example 1: Ensembling, Calibration & Uncertainty

Easily ensemble standard PyTorch models, calibrate them, and decompose their uncertainty to detect Out-Of-Distribution (OOD) data.

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

### Example 2: Variational Inference

Build a Bayesian Neural Network from scratch using our custom layers with the Local Reparameterization Trick.

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

---

## Algorithms & Demos

We implement a wide range of state-of-the-art Bayesian and Ensembling approaches. Check out the interactive demos in the `notebooks/` directory:

| Method | Description |
| :--- | :--- |
| **Deep Ensembles** | Naive yet powerful ensembling of independent networks with explicit uncertainty decomposition. |
| **Monte Carlo Dropout** | Implicit ensembling by keeping dropout active at test time. |
| **Neural Ensemble Search (NES)** | Automatically searches for diverse architectures (NES-RS/NES-RE) using NNI. |
| **NES via Bayesian Sampling** | Extracts diverse subnetworks from a Supernet using Stein Variational Gradient Descent (SVGD). |
| **Variational Inference** | Approximates posterior using Gaussian distributions with the *Local Reparameterization Trick*. |
| **Variational Rényi** | Generalization of VI minimizing $\alpha$-divergence (VR-VI) for better robustness. |
| **Laplace Approximation** | Fits a Gaussian around the MAP estimate using Kronecker-Factored Curvature (K-FAC). |
| **Probabilistic Backprop** | Propagates moments through the network using Assumed Density Filtering (ADF). |

---

## Structure

```text
bensemble/
├── core/                  # Base protocols, ensemble abstractions, and adapters
│   ├── ensemble.py        # Central `Ensemble` class
│   └── member.py          # Adapters for explicit and stochastic models
│
├── layers/                # Bayesian Layers for Variational Inference
│   ├── linear.py          # Bayesian Linear layer
│   └── conv.py            # Bayesian Convolution layer
│
├── search/                # Neural Ensemble Search algorithms
│   ├── nes.py             # NES-RS & NES-RE
│   └── bayesian.py        # NES via Bayesian Sampling
│
├── diversity/             # Methods to induce ensemble variation
│   └── dropout.py         # Monte Carlo Dropout wrapper
│
├── uncertainty/           # Uncertainty analysis
│   └── decomposition.py   # Separation of Aleatoric and Epistemic uncertainty
│
├── calibration/           # Post-hoc model calibration tools
│   └── scaling.py         # Temperature Scaling and Vector Scaling
│
└── metrics.py             # Scoring rules: ECE, NLL, Brier Score
```

---

## Development Setup

If you want to contribute to `bensemble` or run tests, we recommend using **uv**.

```bash
# 1. Clone the repository
git clone https://github.com/intsystems/bensemble.git
cd bensemble

# 2. Create and activate virtual environment via uv
uv venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Linting

We use `ruff` to keep code clean:

```bash
ruff check .
ruff format .
```

---

## Authors

Developed by:

- [**Dmitrii Vasilenko**](https://github.com/Dimundel)
- [**Muhammadsharif Nabiev**](https://github.com/mikhmed-nabiev)
- [**Fedor Sobolevsky**](https://github.com/TeoSable)
- [**Vadim Kasyuk**](https://github.com/KasiukVadim)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
