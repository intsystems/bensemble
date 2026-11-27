# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

## [0.3.0] - 2026-11-27

### Changed

- **Rényi loss.** For `VariationalLoss` with `alpha != 1`, replace the scalar
  KL argument with the keyword arguments `log_prior` and `log_posterior`.
  Use `bensemble.utils.sample_with_log_probs` to obtain predictions and both
  log densities from the same Bayesian linear or convolutional weight draws.
  The ordinary ELBO (`alpha == 1`) still accepts a scalar KL.
- **Laplace posterior.** `LaplaceApproximation` now defaults to
  `regularization="exact"`. It scales K-FAC curvature by the training dataset
  size once, then adds the prior precision. For the previous scaling, pass
  `regularization="legacy"`.

### Added

- `TemperatureScaling.fit` and `VectorScaling.fit` accept a custom optimizer.
- Added reproducible classification and regression benchmarks. Install their
  dependencies from the repository root with `pip install -e ".[benchmarks]"`.
- Published the documentation site with API references and refreshed the
  runnable examples for ensemble prediction, inference and search.

### Fixed

- Laplace curvature accounts for nonlinear activations and branches. Laplace
  ensemble sampling also works with a plain `nn.Module`.
- Ensemble prediction and `predict_with_uncertainty` preserve each layer's
  training mode, including after errors. Frozen BatchNorm statistics stay
  fixed. Stochastic ensembles support dropout variants, keep Bayesian layers
  stochastic in `mode="both"` and reject unknown modes.
- Bayesian Monte Carlo prediction draws one `BayesianConv2d` kernel and bias
  per forward pass, preserving spatial correlations through pooling.
- Ensemble selection aligns predictions with labels under shuffled validation.
  NESBS compares candidates on the same batches, including when validation
  applies random transforms.
- Classification search scores mixture NLL in log space so confident errors
  receive their full penalty. Regression selection aligns scalar targets and
  predictions supplied as vectors or columns.
- `VariationalLoss` reduces likelihoods consistently across Monte Carlo
  samples for classification labels and regression targets.
- Temperature and vector scaling use a strong Wolfe line search to improve
  convergence with the default L-BFGS optimizer.
- Classification entropy treats zero probabilities as zero contributions in
  float16. Passing an explicit `eps` retains the previous formula.
- Fixed package-level imports for `bensemble.uncertainty` decomposition
  functions and `bensemble.diversity.MCDropoutEnsembler`.

## [0.2.2] - 2026-07-03

### Added

- `CONTRIBUTING.md` file with contribution guidelines.

### Changed

- Translated docstrings to English across the codebase.
- Refactored Laplace Approximation and Probabilistic Backpropagation (PBP) implementations.
- Fixed and updated internal code comments.

### Removed

- Deleted `todo.md` as task management is moved elsewhere.

---

## [0.2.1] - 2026-05-31

### Added

- Model pruning support for Bayesian models (`prune_model` function).
- `apply_pruning` function for Bayesian layers.

---

## [0.2.0] - 2026-05-06

### Added

- **Ensemble Core API:** Full `Ensemble` framework including `ExplicitMembers`, `StochasticMembers`, `MemberAdapter`, and factory methods.
- **Ensemble Methods:** Implementation and demos for `MCDropoutEnsembler`.
- **Neural Ensemble Search (NES):** Implementation and demos for NES-RS (Random Search), NES-RE (Regularized Evolution), and an initial prototype for NES-BS (Bayesian Sampler) with a base NES class.
- **Evaluation & Metrics:** Functions for Negative Log-Likelihood (NLL), Brier score, Expected Calibration Error (ECE), and reliability diagrams. Added benchmark notebooks.
- **Calibration:** Vector scaling and Temperature scaling implementations with corresponding tests.
- **Uncertainty Estimation:** `predict_with_uncertainty` utility, along with functions for decomposing regression (`decompose_regression_uncertainty`) and classification (`decompose_classification_uncertainty`) uncertainty.
- **Bayesian Layers:** Weight pruning functions in `BaseBayesianLayer`. Added `kl_weight` hyperparameter to `VariationalLoss` and `sigma` property to `GaussianLikelihood`.
- **Documentation & CI:** Massive documentation update (mkdocs), pre-final versions of the tech report and blog post. Added dynamic badges and Codecov secret token in CI.

### Changed

- Refactored uncertainty decomposition by moving it to a dedicated `uncertainty/` folder.
- Refactored search interfaces for NES methods.
- Refactored `BayesianLinear` and `BayesianConv2d` to inherit from a unified `BaseBayesianLayer`.
- Renamed CI configuration `test.yaml` to `test.yml`.
- Standardized the `tests/` folder structure to mirror the `bensemble/` folder.

### Fixed

- Fixed internal issues with Laplace Approximation.
- Fixed annotations in the MC Dropout demo.
- Fixed `predict_with_uncertainty` to correctly set Bayesian layers to train mode, enabling MC sampling.
- Fixed multiple module imports across the library.
- Fixed GitHub language statistics and YAML syntax spacing in CI.
- Fixed Tech Report links and moved calibration documentation from user-guide to api in `mkdocs.yaml`.

### Removed

- Removed `phi` and `Phi` imports and their related tests.
- Removed `EarlyStopping` and the outdated `compute_uncertainty()` from utils.

---

## [0.1.0] - 2026-01-31

### Added

- **Bayesian Inference Core:** Initial implementations of Variational Inference (VI), Probabilistic Backpropagation (PBP), and Laplace Approximation.
- **Bayesian Layers:** Implemented `BayesianLinear` and `BayesianConv2d` layers.
- **Likelihoods & Losses:** Added `GaussianLikelihood` (with `GaussianLL`), `VariationalLoss` (for VRenyi), and `get_total_kl()` utility.
- **Testing:** Initial integration tests for `BayesianLinear`, math tests for VI, PBP, and Laplace. Added basic `conftest.py`.
- **Documentation:** Initial README, project structure, `mkdocs` prototype, and early tech report/blog post drafts. Added introductory notebooks (VI, PBP, KFLA).

### Changed

- Restructured repository layout: moved layers to `bensemble/layers/`, moved `utils.py` to the root `bensemble/`, and cleaned up the general folder layout.
- Modernized packaging: migrated from `setup.py` and `requirements.txt` to `pyproject.toml` (PEP standards).
- Refactored `forward()` in `BayesianLinear`.
- Modified VI to use `kl_weight` normalized by batch size.
- Refactored Probabilistic Backpropagation and Variational Inference implementations.

### Fixed

- Fixed probabilistic backpropagation and core module imports.
- Fixed scaling issues in PBP and removed redundant `sqrt` in Laplace.
- Fixed key error in `laplace_approximation`.

### Removed

- Deleted experimental classes and folders: `variational-renyi` and `variational-ensemble`.
- Removed outdated folders (`src/`, `code/`, `data/`, `figures/`) and duplicate files to clean up the repository.
- Removed excess `sampling` parameter in `BayesianLinear` initialization.
