# Examples

Start with the four numbered tutorials, then explore neural ensemble search.
Every notebook is self-contained, uses synthetic data on CPU, and runs from top
to bottom without downloading datasets. The code needs only the library's runtime
dependencies; JupyterLab provides the interactive notebook interface.

## Setup

From the repository root, create and activate an environment, then install the
local package and JupyterLab:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e . jupyterlab
python -m jupyterlab examples/
```

With an existing `uv` environment, the equivalent installation is:

```bash
uv pip install -e . jupyterlab
uv run --no-sync python -m jupyterlab examples/
```

Select a kernel from the environment where you installed `bensemble`. Restart
the kernel and run all cells for each notebook. Imports use the installed package,
so no `sys.path` edits or particular working directory are required.

## Learning path

| Notebook | What it teaches | Approximate CPU budget |
| --- | --- | --- |
| [1. Ensemble basics](01_ensemble_basics.ipynb) | Train independent classifiers, average probabilities, calibrate on held-out data, decompose entropy | Under 1 minute |
| [2. MC Dropout](02_mc_dropout.ipynb) | Turn one dropout network into a stochastic ensemble | Under 1 minute |
| [3. Variational inference](03_variational_inference.ipynb) | Bayesian layers, ELBO/Rényi objectives with multiple samples, observation noise | Under 1 minute |
| [4. Laplace approximation](04_laplace.ipynb) | Fit curvature around a trained classifier and sample an explicit ensemble | Under 1 minute |
| [NES-RS](search/nes_rs.ipynb) | Random architecture search and greedy ensemble selection | Under 1 minute |
| [NES-RE](search/nes_re.ipynb) | Ensemble-aware evolutionary search | Under 1 minute |
| [NES via Bayesian sampling](search/nes_bayesian_sampling.ipynb) | Validation-loss posterior, MC selection, and SVGD-inspired refinement | Under 1 minute |

Budgets are rough guides for a recent CPU after imports; hardware and package
versions affect runtime. Training budgets are intentionally small. Python and
PyTorch seeds are fixed, but exact results can differ between software versions.

## Conventions

- Train, validation, and test observations are disjoint. Fit preprocessing on train
  only; use validation for calibration or candidate selection; report final
  metrics on test. The Laplace demo selects the sampling temperature by validation
  NLL. The regression demo reserves validation for optional tuning.
- Classification mixtures average each member's softmax probabilities.
  `Ensemble` averages raw outputs by default, so use `predict_members` explicitly
  when your models return logits. Regression ensembles can average outputs directly.
- Rényi training (`alpha != 1`) uses `sample_with_log_probs` to pair predictions
  with weight log densities. A scalar KL is supported only for the ordinary ELBO.
  Notebook 3 handles both paths; change `ALPHA` and rerun all cells.
- Regression bands show model variation, or model variation plus learned noise
  where stated. They do not establish calibrated 95% coverage. OOD uncertainty
  growth and calibration improvements are observations to investigate, not promises.
- Small data and training definitions are repeated intentionally to keep each
  notebook independently runnable and readable.

For longer experiments and comparisons across seeds, use [benchmarks](../benchmarks/README.md).
For methods without a dedicated notebook, including PBP, see the
[algorithm documentation](../docs/docs/algorithms/pbp.md).

## Checks and contributions

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_examples.py -q
ruff check examples/ tests/test_examples.py
ruff format --check examples/ tests/test_examples.py
```

The smoke tests execute the actual Python cells in order, in a fresh process per
notebook with its directory as the working directory. They render plots with the
noninteractive Agg backend, check data separation and predictions, and enforce a
timeout. No Jupyter server is required for this check. It is part of the normal
test suite and runs in CI; notebook linting also runs in CI.

Keep code cells plain Python, without installation commands or notebook magics.
Before committing, clear outputs and execution counts; generated plots and local
kernel metadata should not be stored in the notebooks. Keep the guide, relative
notebook links, and links from the main README and blog up to date when renaming.
