# Benchmarks

This folder contains reproducible, cross-method validation scripts for `bensemble` that run every supported ensembling/UQ method through the same pipeline under matched conditions.

- `classification_benchmark.py` — 8 methods (Single Net, Deep Ensemble, MC Dropout, VI, Laplace K-FAC, NESBS, NES-RS, NES-RE) on CIFAR-10 (ID), a noise-shifted CIFAR-10 (distribution shift), and SVHN (OOD detection).
- `regression_benchmark.py` — PBP vs. VI vs. Laplace vs. a plain MAP baseline on UCI regression datasets (Yacht, Energy, Concrete, Power Plant), since PBP only supports regression and can't run in the classification pipeline above.
