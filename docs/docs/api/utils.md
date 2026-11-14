# Utilities

Helpers for stochastic inference, model pruning and Gaussian moment
propagation.

`sample_with_log_probs` returns predictions and matching prior/posterior log
densities for Rényi training. See the [Rényi guide](../algorithms/variational-renyi.md)
for usage; `predict_with_uncertainty` returns predictive mean and standard
deviation and does not supply weight densities.

::: bensemble.utils
