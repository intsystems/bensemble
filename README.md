# 🚧 Under Active Refactoring 🚧
The library is currently undergoing a major rewrite. The API might be unstable. Check back soon!

---

# 😎 Bensemble: Bayesian Multimodeling Project

[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](tests/)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://intsystems.github.io/bensemble/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://img.shields.io/badge/uv-managed-a6c489)](https://github.com/astral-sh/uv)

**Bensemble** is a library for Bayesian Deep Learning which integrates established  methods for neural network ensembling and uncertainty quantification. Bensemble provides building blocks that slot directly into your existing PyTorch workflows.

---

##  Key Resources

| Resource | Description |
| :--- | :--- |
| 📘 **[Documentation](https://intsystems.github.io/bensemble/)** | Full API reference and user guides. |
| 📝 **[Tech Report](https://github.com/intsystems/bensemble/blob/master/paper/techreport_draft/bensemble_tech_report.pdf)** | In-depth technical details and theoretical background. |
| ✍️ **[Blog Post](https://mikhmed-nabiev.github.io/bayesian-multimodelling.html)** | Summary of the project and motivation. |
| 📊 **[Benchmarks](https://github.com/intsystems/bensemble/blob/master/notebooks/benchmark.ipynb)** | Comparison of methods on standard datasets. |

---

## Features

- **PyTorch-Native**: All layers and methods compatible with standart PyTorch.
- **Modularity**: BayesianLinear, BayesianConv2d with built-in Local Reparameterization Trick (LRT).
- **Core Bayesian Methods**: Implements canonical algorithms from Variational Inference to Scalable Laplace approximations.
- **Modern Stack**: Built with `uv`, fully typed, and tested.

---

## Installation

You can install `bensemble` using pip:

```bash
pip install bensemble
```

Or, if you prefer using [uv](https://github.com/astral-sh/uv) for lightning-fast installation:

```bash
uv pip install bensemble
```

---

##  Quick Start

Build a Bayesian Neural Network using our layers and write a standard PyTorch training loop.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import building blocks
from bensemble.layers import BayesianLinear
from bensemble.objectives import VariationalLoss, GaussianLikelihood
from bensemble.utils import get_total_kl

# 0. Prepare Dummy Data
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)

# 1. Define Model using Bayesian Layers
model = nn.Sequential(
    BayesianLinear(10, 50, prior_sigma=1.0),
    nn.ReLU(),
    BayesianLinear(50, 1, prior_sigma=1.0)
)

# 2. Define Objectives (Likelihood + Divergence)
likelihood = GaussianLikelihood()
criterion = VariationalLoss(likelihood, alpha=1.0)

optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)

# 3. Train Model
model.train()
for x, y in train_loader:
    optimizer.zero_grad()
    
    preds = model(x)
    kl = get_total_kl(model)
    
    loss = criterion(preds, y, kl)
    
    loss.backward()
    optimizer.step()

# 4. Predict
mean, std = likelihood.predict(model(x_test))
```

---

## Development Setup

If you want to contribute to `bensemble` or run tests, we recommend using **uv** to manage the environment.

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

---

##  Algorithms & Demos

We have implemented four distinct approaches. Check out the interactive demos for each:

| Method | Description | Demo |
| :--- | :--- | :--- |
| **Variational Inference** | Approximates posterior using Gaussian distributions using [*Local Reparameterization Trick*](https://arxiv.org/abs/1506.02557) | [Open Notebook](https://github.com/intsystems/bensemble/blob/master/notebooks/pvi_demo.ipynb) |
| **Laplace Approximation** | Fits a Gaussian around the MAP estimate using Kronecker-Factored Curvature (K-FAC). | [Open Notebook](https://github.com/intsystems/bensemble/blob/master/notebooks/laplace_demo.ipynb) |
| **Variational Rényi** | Generalization of VI minimizing $\alpha$-divergence (Rényi). | [Open Notebook](https://github.com/intsystems/bensemble/blob/master/notebooks/variatinal_renyi_demo.ipynb) |
| **Probabilistic Backprop** | Propagates moments through the network using Assumed Density Filtering (ADF). | [Open Notebook](https://github.com/intsystems/bensemble/blob/master/notebooks/pbp_probabilistic_backpropagation_test.ipynb) |

---

##  Development & Testing

The library is covered by a comprehensive test suite to ensure reliability.

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

##  Authors

Developed by:
* [**Fedor Sobolevsky**](https://github.com/TeoSable)
* [**Muhammadsharif Nabiev**](https://github.com/mikhmed-nabiev)
* [**Dmitrii Vasilenko**](https://github.com/Dimundel)
* [**Vadim Kasyuk**](https://github.com/KasiukVadim)

---

##  License


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



