# NES via Bayesian Sampling

To reduce the prohibitive computational cost of standard NES, one can use Neural Ensemble Search via Bayesian Sampling.

The original method trains a Supernet with weight sharing and learns a variational posterior over architectures. `bensemble` implements a discrete, pool-based version of that idea instead: `NESBayesianSampler` draws `pool_size` architectures from the `SearchSpace`, trains each one independently with the user's `train_fn`, and scores it on a validation set. The scores define a posterior over the pool,

$$
p(\mathcal{A}_i \mid \mathcal{D}) \propto \exp\!\left(-\frac{s_i - \min_j s_j}{T}\right),
$$

where $s_i$ is the validation loss of candidate $i$ and $T$ is the `temperature`.

Ensemble members are then selected from the pool in one of two ways:

- **Monte-Carlo Sampling** (`sample_mc`): draw `ensemble_size` candidates from the posterior.
- **SVGD-inspired sampling** (`sample_svgd`): a greedy, particle-style selection over the pool. Each candidate's posterior probability is traded off against a repulsion term measuring how similar its validation predictions are to those of the members already chosen, so the selected set is pushed towards architectures that disagree with each other.

$$
q^* = \arg\min_{q\in\mathcal{Q}} \text{KL}(q\|p) + n\delta\mathbb{E}_{x, x' \sim q}[k(x, x')]
$$

The objective above is the one the original paper optimizes with Stein Variational Gradient Descent; here it motivates the repulsion heuristic rather than being solved exactly.

---

Yao Shu et al. [*"Neural Ensemble Search via Bayesian Sampling"*](https://arxiv.org/abs/2109.02533) (2022)
