# Welcome to Bensemble 😎

**Bensemble** is a lightweight, modular library for Bayesian Deep Learning and Neural Network Ensembling in PyTorch.

Unlike other frameworks that force you into rigid, black-box training loops, Bensemble provides **composable building blocks** (layers, losses, metrics, and search algorithms) that slot directly into your existing pure PyTorch workflows.

## Why Bensemble?

- **Unified API**: A single `Ensemble` interface for Deep Ensembles, MC Dropout, and advanced Neural Ensemble Search (NAS).
- **Uncertainty Analytics**: Built-in tools to rigorously separate *Aleatoric* (data noise) and *Epistemic* (model ignorance) uncertainty.
- **Calibration**: Easy-to-use Temperature and Vector scaling to fix overconfident neural networks.
- **PyTorch-Native**: Train your models exactly how you want. We only step in when it's time to ensemble and evaluate.

## Quick Links

- [🚀 Getting Started](getting-started.md)
- [🧠 Basic Concepts](user-guide/basic-concepts.md)
- [📊 Uncertainty Guide](user-guide/uncertainty-analysis.md)
- [⚙️ API Reference](api/core.md)
