# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

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
