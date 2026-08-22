# Benchmarks

This folder contains reproducible, cross-method validation scripts for bensemble that run every supported ensembling/UQ method through the same pipeline under matched conditions.

## Running benchmarks

> **Note:** Run all commands from the **root directory** of the repository.

### Option 1: Using `uv`

Run classification benchmark:

```bash
uv run --extra benchmarks python benchmarks/classification_benchmark.py
```

Run regression benchmark:

```bash
uv run --extra benchmarks python benchmarks/regression_benchmark.py
```

### Option 2: Using pip and venv

1. Create and activate venv

   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```

2. Install dependencies

   ```bash
   pip install -e ".[benchmarks]"
   ```

3. Run classification or regression benchmark:

   ```bash
   python benchmarks/classification_benchmark.py
   ```

   ```bash
   python benchmarks/regression_benchmark.py
   ```

## Classification benchmark

`classification_benchmark.py`

Evaluates all classification / uncertainty estimation methods on:

- **In-distribution:** CIFAR-10
- **Distribution shift:** Gaussian-noised CIFAR-10
- **Out-of-distribution:** SVHN

Implemented methods:

| Method | ID Acc | ECE ↓ | OOD AUROC ↑ |
| ------ | -----: | ----: | ----------: |
| Single Net | 92.72% | 0.0569 | 0.500 |
| Deep Ensemble | 93.80% | 0.0267 | 0.898 |
| MC Dropout | 92.90% | 0.0525 | 0.899 |
| Laplace (K-FAC) | 92.78% | 0.0550 | 0.848 |
| VI (ELBO) | 92.91% | **0.0197** | 0.749 |
| NES-BS | 93.73% | 0.0239 | **0.907** |
| NES-RS | **93.94%** | 0.0237 | **0.907** |
| NES-RE | 93.82% | 0.0243 | 0.891 |
