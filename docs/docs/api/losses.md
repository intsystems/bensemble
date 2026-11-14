# Losses

Objectives for training variational Bayesian models.

## Variational Loss

For `alpha=1`, pass the analytic KL as the third argument. For `alpha != 1`,
generate matched draws with `bensemble.utils.sample_with_log_probs` and pass
`log_prior` and `log_posterior` as keyword arguments. A scalar KL alone raises
`ValueError` on the Rényi path. See the [training example and migration
notes](../algorithms/variational-renyi.md).

::: bensemble.losses.VariationalLoss

## Gaussian Likelihood

::: bensemble.losses.GaussianLikelihood
