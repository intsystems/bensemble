# Basic Concepts

The main goal of **Bensemble** is to provide a unified interface for different ensembling and Bayesian methods.

### The Ensemble Container

Everything revolves around the `Ensemble` class. It acts as a manager for multiple "members" (individual neural networks).

- **Explicit Ensembles**: A collection of different models (e.g., from NAS or Deep Ensembles).
- **Implicit Ensembles**: A single model that behaves like an ensemble (e.g., MC Dropout or Bayesian layers).

Regardless of the source, an `Ensemble` always returns a tensor of shape `[M, Batch, Output]`, where `M` is the number of ensemble members.
