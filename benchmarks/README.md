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

## Reproducibility

Both scripts fix every random seed and restrict PyTorch to deterministic kernels, so repeated runs on the same machine and software stack produce identical results.

On CUDA, one more setting is needed and has to come from the environment:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Without it, `torch.use_deterministic_algorithms(True)` raises as soon as a cuBLAS reduction runs.

Results still differ across GPU architectures, CUDA/cuDNN versions and PyTorch releases: floating-point addition is not associative, so a different kernel sums in a different order. The numbers reported here were produced on the hardware listed under [Runtime](#runtime); expect small deviations elsewhere.

## Requirements

- **Python:** 3.10–3.13
- **OS:** Linux, Windows, and macOS
- **Hardware:**
  - **CPU:** Supported (sufficient for regression, but slow for full classification).
  - **CUDA:** Recommended for the classification benchmark.

## Runtime

Approximate total runtime on an **NVIDIA RTX A4000**:

| Benchmark      | Total runtime |
|----------------|--------------:|
| Classification | 4h 50min      |
| Regression     | 1h 15min      |

## Data

Datasets are downloaded automatically upon running the scripts. No manual preparation is needed.

### Datasets Used

- **Classification (via `torchvision.datasets`):**
  - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — In-Distribution
  - [SVHN](http://ufldl.stanford.edu/housenumbers/) — Out-of-Distribution

- **Regression (via `ucimlrepo` / UCI Archive):**
  - [Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency) — UCI ID: 242
  - [Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) — UCI ID: 165
  - [Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) — UCI ID: 294
  - [Yacht Hydrodynamics](https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data)
  
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

## Regression benchmark

`regression_benchmark.py`

Evaluates all regression / uncertainty estimation methods on:

- **UCI datasets:** Yacht (N=308), Energy (N=768), Concrete (N=1030), Power Plant (N=9568)
- **Protocol:** five random train/test splits per dataset, reported as mean ± std

Implemented methods:

RMSE ↓

| Method | Yacht | Energy | Concrete | Power Plant |
| ------ | -----------: | -----------: | -----------: | -----------: |
| PBP | 1.931 ± 0.427 | 2.880 ± 0.228 | 6.561 ± 0.518 | 4.142 ± 0.163 |
| VI (ELBO) | 2.955 ± 0.886 | 2.698 ± 0.263 | 6.066 ± 0.385 | 4.149 ± 0.142 |
| Laplace (K-FAC) | 1.036 ± 0.172 | 0.975 ± 0.086 | 5.607 ± 0.448 | **4.026 ± 0.151** |
| MAP (baseline) | **0.929 ± 0.161** | **0.844 ± 0.101** | **5.554 ± 0.438** | 4.027 ± 0.148 |

NLPD ↓

| Method | Yacht | Energy | Concrete | Power Plant |
| ------ | -----------: | -----------: | -----------: | -----------: |
| PBP | 3.068 ± 0.006 | 2.776 ± 0.022 | 3.448 ± 0.031 | 3.240 ± 0.009 |
| VI (ELBO) | 2.708 ± 0.130 | 2.476 ± 0.069 | 3.225 ± 0.056 | 2.847 ± 0.028 |
| Laplace (K-FAC) | 1.651 ± 0.112 | 1.439 ± 0.124 | 3.145 ± 0.083 | 2.818 ± 0.034 |
| MAP (baseline) | **1.410 ± 0.203** | **1.267 ± 0.128** | **3.138 ± 0.082** | **2.818 ± 0.033** |

NLPD is not directly comparable across methods: MAP and Laplace fit the noise on a validation split, VI and PBP infer it themselves. Laplace adds epistemic variance on top of that fitted noise, so it trails MAP on the smaller datasets.
