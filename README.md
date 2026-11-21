# Bensemble: Modular Bayesian Deep Learning & Ensembling

[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/intsystems/bensemble/actions/workflows/test.yml/badge.svg)](https://github.com/intsystems/bensemble/actions)
[![Codecov](https://codecov.io/gh/intsystems/bensemble/branch/master/graph/badge.svg)](https://codecov.io/gh/intsystems/bensemble)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://intsystems.github.io/bensemble/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![JOSS](https://joss.theoj.org/papers/0d7c1f90dc95cc1a9c913a29e7922a4d/status.svg)](https://joss.theoj.org/papers/0d7c1f90dc95cc1a9c913a29e7922a4d)

**Bensemble** is a PyTorch library for combining neural networks, estimating predictive uncertainty, and calibrating predictions.

---

## Key Resources

| Resource | Description |
| :--- | :--- |
| 📘 **[Documentation](https://intsystems.github.io/bensemble/)** | Full API reference and user guides. |
| 📝 **[Tech Report](https://github.com/intsystems/bensemble/blob/master/paper/bensemble_tech_report.pdf)** | In-depth technical details and theoretical background. |
| ✍️ **[Blog Post](https://github.com/intsystems/bensemble/blob/master/blog/bensemble-blogpost.md)** | Summary of the project and motivation. |
| 📊 **[Benchmarks](https://github.com/intsystems/bensemble/blob/master/benchmarks)** | Comparison of methods on standard datasets. |

---

## Features

- Combine independently trained PyTorch models or draw predictions from one model with MC dropout using `Ensemble`.
- Select ensembles with random search (NES-RS), evolutionary search (NES-RE), or Bayesian sampling (NES-BS).
- Split predictive uncertainty into aleatoric and epistemic components for classification and regression.
- Calibrate classifiers with temperature or vector scaling. Evaluate predictions with expected calibration error (ECE), Brier score, and negative log-likelihood.

---

## Installation

```bash
pip install bensemble
```

With [uv](https://github.com/astral-sh/uv), run `uv pip install bensemble`.

---

## Quick Start

### Combine trained classifiers

`models` is a list of trained PyTorch classifiers that return logits for the same classes in the same order. `x` is a batch of inputs.

```python
import torch

from bensemble.core.ensemble import Ensemble
from bensemble.uncertainty import decompose_classification_uncertainty

ensemble = Ensemble.from_models(models).eval()
with torch.no_grad():
    member_probs = ensemble.predict_members(x).softmax(dim=-1)  # [members, batch, classes]
    probs = member_probs.mean(dim=0)
    total, aleatoric, epistemic = decompose_classification_uncertainty(member_probs)
```

For a runnable example with training and calibration, see [Getting Started](https://intsystems.github.io/bensemble/getting-started/).

### Variational inference

Build a Bayesian Neural Network from scratch using our custom layers with the Local Reparameterization Trick.

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

For Rényi training (`alpha != 1`), use matched weight draws and log densities:

```python
from bensemble.utils import sample_with_log_probs

criterion = VariationalLoss(likelihood, alpha=0.5)
optimizer.zero_grad()
predictions, log_p, log_q = sample_with_log_probs(model, x, num_samples=8)
loss = criterion(predictions, y, log_prior=log_p, log_posterior=log_q)
loss.backward()
optimizer.step()
```

A scalar KL is accepted only for the ordinary ELBO (`alpha=1`). See the
[Rényi guide](docs/docs/algorithms/variational-renyi.md) for the objective and
minibatch scaling.

---

## Algorithms & Demos

Start with the [examples guide](examples/README.md): four ordered tutorials cover
ensembling, MC Dropout, variational inference, and Laplace; `examples/search/`
contains the three NES methods. All seven notebooks use synthetic data on CPU
and include instructions for running them from a fresh kernel.

| Method | Description |
| :--- | :--- |
| **[Deep Ensembles](examples/01_ensemble_basics.ipynb)** | Train independent classifiers, calibrate on held-out data, and decompose uncertainty. |
| **[Monte Carlo Dropout](examples/02_mc_dropout.ipynb)** | Implicit ensembling by keeping dropout active at test time. |
| **Neural Ensemble Search ([NES-RS](examples/search/nes_rs.ipynb) / [NES-RE](examples/search/nes_re.ipynb))** | Search architectures and select an ensemble on a separate validation set. |
| **[NES via Bayesian Sampling](examples/search/nes_bayesian_sampling.ipynb)** | Select from independently trained candidates using a validation-loss posterior and SVGD-inspired refinement. |
| **[Variational Inference](examples/03_variational_inference.ipynb)** | Learn a Gaussian approximation with the *Local Reparameterization Trick*. |
| **[Variational Rényi](examples/03_variational_inference.ipynb)** | Train with matched weight draws and per-sample prior/posterior log densities. |
| **[Laplace Approximation](examples/04_laplace.ipynb)** | Estimate curvature around a trained network and sample an ensemble. |
| **[Probabilistic Backprop](docs/docs/algorithms/pbp.md)** | Propagate moments using Assumed Density Filtering (ADF); see the algorithm guide. |

---

## Structure

```text
bensemble/
├── core/                  # Base protocols, ensemble abstractions, and adapters
│   ├── ensemble.py        # Central `Ensemble` class
│   ├── member.py          # Adapters for explicit and stochastic models
│   └── types.py           # Protocols: Predictor, KLProvider, PosteriorSource
│
├── layers/                # Bayesian Layers for Variational Inference
│   ├── base.py            # BaseBayesianLayer: KL divergence, SNR-based pruning
│   ├── linear.py          # Bayesian Linear layer
│   └── conv.py            # Bayesian Convolution layer
│
├── methods/               # Standalone posterior-approximation engines
│   ├── laplace_approximation.py    # Laplace with K-FAC curvature
│   └── probabilistic_backpropagation.py  # PBP (ADF-based)
│
├── search/                # Neural Ensemble Search algorithms
│   ├── nes.py              # NES-RS & NES-RE (RandomSearcher, EvolutionarySearcher)
│   ├── bayesian.py         # NESBS (discrete, SVGD-inspired sampler)
│   ├── selection.py        # Greedy forward selection, scoring criteria
│   └── space.py            # SearchSpace protocol
│
├── diversity/              # Methods to induce ensemble variation
│   └── dropout.py          # Monte Carlo Dropout wrapper
│
├── uncertainty/            # Uncertainty analysis
│   └── decomposition.py    # Separation of Aleatoric and Epistemic uncertainty
│
├── calibration/            # Post-hoc model calibration tools
│   └── scaling.py          # Temperature Scaling and Vector Scaling
│
├── losses.py                # VariationalLoss, GaussianLikelihood
├── utils.py                 # get_total_kl, predict_with_uncertainty, etc.
└── metrics.py                # Scoring rules: ECE, NLL, Brier Score
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
- [**Vadim Kasiuk**](https://github.com/KasiukVadim)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
