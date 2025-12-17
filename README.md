# 😎 Bensemble: Bayesian Multimodeling Project

[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](tests/)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://intsystems.github.io/bensemble/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://img.shields.io/badge/uv-managed-a6c489)](https://github.com/astral-sh/uv)

**Bensemble** is a comprehensive comparative study and a production-ready library for Bayesian Deep Learning. It integrates established  methods for neural network ensembling and uncertainty quantification under a unified PyTorch interface.

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

- **Unified API**: All methods share a consistent `fit` / `predict` interface (Scikit-learn style).
- **Core Bayesian Methods**: Implements canonical algorithms from Variational Inference to Scalable Laplace approximations.
- **Modern Stack**: Built with `uv`, fully typed, and tested with **98% code coverage**.

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

##  Quick Start

Here is how to turn a standard PyTorch model into a Bayesian one using **Variational Inference**:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bensemble import VariationalEnsemble

# 0. Prepare dummy data
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 1. Define your standard PyTorch model
model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))

# 2. Wrap it with Bensemble
# auto_convert=True automatically replaces Linear layers with BayesianLinear
ensemble = VariationalEnsemble(model, auto_convert=True)

# 3. Train (Fit)
history = ensemble.fit(train_loader, epochs=50)

# 4. Predict with Uncertainty
# Returns mean prediction and standard deviation (uncertainty)
X_test = torch.randn(5, 10)
mean, std = ensemble.predict(X_test, n_samples=100)

print(f"Prediction: {mean[0].item():.4f} ± {std[0].item():.4f}")
```

---

##  Algorithms & Demos

We have implemented four distinct approaches. Check out the interactive demos for each:

| Method | Description | Demo |
| :--- | :--- | :--- |
| **Variational Inference** | Uses "Bayes By Backprop" with Local Reparameterization Trick. | [Open Notebook](https://github.com/intsystems/bensemble/blob/master/notebooks/pvi_demo.ipynb) |
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



