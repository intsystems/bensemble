# Variational Inference with Rényi Divergence (VR)

This method generalizes the standard ELBO using $\alpha$-Rényi divergence.

For each draw $w_k \sim q_\theta(w)$, the prediction and both weight densities must
use the same weights. The Monte Carlo objective is:

$$
\mathcal{L}_{\text{VR}}(\theta, \alpha) = -\frac{1}{1-\alpha} \log \frac{1}{K} \sum_{k=1}^K \left( \frac{p(\mathcal{D}, w_k)}{q_\theta(w_k)} \right)^{1-\alpha}
$$

`sample_with_log_probs` draws one set of weights per forward pass for
`BayesianLinear` and `BayesianConv2d`. The draw is shared across the whole batch
and repeated calls to the same layer. It returns predictions and differentiable
log densities in matching sample order:

```python
from bensemble.losses import VariationalLoss
from bensemble.utils import sample_with_log_probs

criterion = VariationalLoss(likelihood, alpha=0.5)
optimizer.zero_grad()
predictions, log_prior, log_posterior = sample_with_log_probs(
    model, x_train, num_samples=8
)
loss = criterion(
    predictions, y_train, log_prior=log_prior, log_posterior=log_posterior
)
loss.backward()
optimizer.step()
```

Predictions have shape `(K, batch, ...)`; each log-density tensor has shape `(K,)`
and sums over all sampled weights and biases. Use an element-wise or summed NLL
likelihood, such as `CrossEntropyLoss(reduction="none")`, rather than a mean NLL.
The helper preserves layer training flags. Other modules retain their usual
behavior; additional random mechanisms such as dropout are not included in the
returned weight densities.

Ordinary Bayesian forwards use local reparameterization. Those activation draws
do not correspond to one shared weight draw, so they cannot be paired with
separately sampled weight densities for this objective.

At `alpha=1`, `VariationalLoss` still accepts the analytic KL as its third argument
for the ordinary ELBO. Alternatively, pass sampled log densities to obtain the
Monte Carlo ELBO on the same draws as Rényi. With one draw the Monte Carlo
objective is independent of `alpha`; use multiple draws to distinguish them.

For equally sized minibatches, `num_batches=M` uses
$z_k = M\log p(\mathcal{B}\mid w_k) + \log p(w_k) - \log q_\theta(w_k)$
inside the log-mean-exp, then divides the loss by $M$. This estimates the full-data
objective at the same scale as the existing minibatch ELBO. It is a minibatch
approximation, not an unbiased estimate of the full-data Rényi objective.
`num_batches=1` uses the supplied data as the full batch. Non-unit `kl_weight`
scales the log prior/posterior ratio and changes the objective.

**Migration:** calls with `alpha != 1` and only a scalar KL now raise `ValueError`.
Earlier versions substituted that average KL for every draw inside the nonlinear
objective. Replace those calls with matched samples as above; existing
`alpha=1` calls do not need to change.

---

Yingzhen Li, Richard E. Turner [*"Rényi Divergence Variational Inference"*](https://arxiv.org/abs/1602.02311) (2016)
