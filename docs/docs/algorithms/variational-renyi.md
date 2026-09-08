# Variational Inference with Rényi Divergence (VR)

This method generalizes the standard ELBO using $\alpha$-Rényi divergence.

It uses the same Bayesian layers as [Variational Inference](variational-inference.md), so weights are still sampled with the Local Reparameterization Trick; only the objective changes. Pass `alpha` to `VariationalLoss` and feed it $K$ stochastic forward passes stacked along the first dimension. The objective is defined as:

$$
\mathcal{L}_{\text{VR}}(\theta, \alpha) = -\frac{1}{1-\alpha} \log \frac{1}{K} \sum_{k=1}^K \left( \frac{p(\mathcal{D}, w_k)}{q_\theta(w_k)} \right)^{1-\alpha}
$$

The parameter $\alpha$ (default 1.0) controls the bias-variance trade-off, allowing for more robust posterior approximations compared to standard Kullback-Leibler divergence.

---

Yingzhen Li, Richard E. Turner [*"Rényi Divergence Variational Inference"*](https://arxiv.org/abs/1602.02311) (2016)
