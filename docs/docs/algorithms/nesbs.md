# NES via Bayesian Sampling

To reduce the prohibitive computational cost of standard NES, one can use Neural Ensemble Search via Bayesian Sampling.

It utilizes training a Supernet with uniform path sampling to share weights across different model architectures. A variational posterior over architectures $p_\alpha(\mathcal{A}) \approx p(\mathcal{A}|\mathcal{D})$ is learned via ELBO minimization.

Ensemble member architectures can then be sampled from the variational posterior using two methods:

- **Monte-Carlo Sampling**: Simple random sampling from the posterior.
- **SVGD-RD**: Stein Variational Gradient Descent with Regularized Diversity. This uses controlled optimization of the set of architectures with the following objective:

$$
q^* = \arg\min_{q\in\mathcal{Q}} \text{KL}(q\|p) + n\delta\mathbb{E}_{x, x' \sim q}[k(x, x')]
$$

This repulsive force mathematically ensures that the sampled architectures are highly diverse.

---

Yao Shu et al. [*"Neural Ensemble Search via Bayesian Sampling"*](https://arxiv.org/abs/2109.02533) (2022)
