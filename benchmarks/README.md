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
| Classification | 5h 20min      |
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
- **Distribution shift:** CIFAR-10 with Gaussian noise, σ = 0.1 in `[0, 1]` pixel space
- **Out-of-distribution:** SVHN
- **Protocol:** five seeds, reported as mean ± std

Implemented methods:

| Method | ID Acc ↑ | ECE ↓ | Shift Acc ↑ | OOD AUROC ↑ |
| ------ | -----------: | -----------: | -----------: | -----------: |
| Single Net | 92.74 ± 0.15% | 0.0566 ± 0.0014 | 29.25 ± 2.54% | 0.5000 ± 0.0000 |
| Deep Ensemble | 93.68 ± 0.04% | 0.0256 ± 0.0005 | 27.43 ± 1.00% | 0.9015 ± 0.0074 |
| MC Dropout | 92.65 ± 0.14% | 0.0552 ± 0.0011 | 27.28 ± 1.04% | 0.8734 ± 0.0148 |
| Laplace (K-FAC) | 92.74 ± 0.15% | 0.0565 ± 0.0014 | **29.30 ± 2.15%** | 0.8634 ± 0.0237 |
| VI (ELBO) | 92.87 ± 0.08% | **0.0139 ± 0.0015** | 26.85 ± 1.60% | 0.7533 ± 0.0267 |
| NES-BS | **93.77 ± 0.17%** | 0.0249 ± 0.0015 | 27.55 ± 0.97% | 0.8926 ± 0.0078 |
| NES-RS | 93.77 ± 0.09% | 0.0254 ± 0.0011 | 29.20 ± 0.28% | 0.8879 ± 0.0088 |
| NES-RE | 93.76 ± 0.06% | 0.0253 ± 0.0007 | 29.21 ± 2.87% | **0.9062 ± 0.0082** |

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
