# 😎 Bensemble: Bayesian Multimodeling Project

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](tests/)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://intsystems.github.io/bensemble/)

**Bensemble** is a comprehensive comparative study and a production-ready library for Bayesian Deep Learning. It integrates state-of-the-art methods for neural network ensembling and uncertainty quantification under a unified PyTorch interface.

---

## 📚 Key Resources

| Resource | Description |
| :--- | :--- |
| 📘 **[Documentation](https://intsystems.github.io/bensemble/)** | Full API reference and user guides. |
| 📝 **[Tech Report](https://github.com/intsystems/bensemble/blob/master/paper/techreport_draft/bensemble_tech_report.pdf)** | In-depth technical details and theoretical background. |
| ✍️ **[Blog Post](https://mikhmed-nabiev.github.io/bayesian-multimodelling.html)** | Summary of the project and motivation. |
| 📊 **[Benchmarks](https://github.com/intsystems/bensemble/blob/master/notebooks/benchmark.ipynb)** | Comparison of methods on standard datasets. |

---

## 🚀 Features

- **Unified API**: All methods share a consistent `fit` / `predict` interface (Scikit-learn style).
- **4 SOTA Algorithms**: From Variational Inference to Scalable Laplace approximations.
- **Model Agnostic**: Works with standard `torch.nn.Module` architectures.
- **Modern Stack**: Built with `uv`, fully typed, and tested with **98% code coverage**.

---

## 📦 Installation

We recommend using a virtual environment.

### For Users (pip)
```bash
pip install .
```

### For Developers (uv)
We use [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.

```bash
# 1. Install uv
pip install uv

# 2. Create virtual environment
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install in editable mode with dev tools
uv pip install -e ".[dev]"
```

---

## ⚡ Quick Start

Here is how to turn a standard PyTorch model into a Bayesian one using **Variational Inference**:

```python
import torch
import torch.nn as nn
from bensemble.methods import VariationalEnsemble

# 1. Define your standard PyTorch model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

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

## 🧠 Algorithms & Demos

We have implemented four distinct approaches. Check out the interactive demos for each:

| Method | Description | Demo |
| :--- | :--- | :--- |
| **Variational Inference** | Uses "Bayes By Backprop" with Local Reparameterization Trick. | [Open Notebook](https://github.com/intsystems/bensemble/blob/master/notebooks/pvi_demo.ipynb) |
| **Laplace Approximation** | Fits a Gaussian around the MAP estimate using Kronecker-Factored Curvature (K-FAC). | [Open Notebook](https://github.com/intsystems/bensemble/blob/master/notebooks/laplace_demo.ipynb) |
| **Variational Rényi** | Generalization of VI minimizing $\alpha$-divergence (Rényi). | [Open Notebook](https://github.com/intsystems/bensemble/blob/master/notebooks/variatinal_renyi_demo.ipynb) |
| **Probabilistic Backprop** | Propagates moments through the network using Assumed Density Filtering (ADF). | [Open Notebook](https://github.com/intsystems/bensemble/blob/master/notebooks/pbp_probabilistic_backpropagation_test.ipynb) |

---

## 🛠 Development & Testing

We take quality seriously. The library is covered by a comprehensive test suite to ensure reliability.

### Run Tests
```bash
pytest tests/
```

### Check Coverage
To generate the HTML coverage report:
```bash
pytest --cov=bensemble --cov-report=html tests/
# Open htmlcov/index.html in your browser
```

### Linting
We use `ruff` to keep code clean:
```bash
ruff check .
ruff format .
```

---

## 👥 Authors

Developed by the **Intelligent Systems Team**:
* **Fedor Sobolevskiy**
* **Mukhamadsharif Nabiev**
* **Dmitrii Vasilenko**
* **Vadim Kasyuk**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.