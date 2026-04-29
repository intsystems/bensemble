# Practical Variational Inference

Variational Inference approximates the true posterior with a diagonal Gaussian $q_\theta(w)$. The objective is to minimize the variational free energy (Evidence Lower Bound, ELBO):

$$
\mathcal{L}(\theta) = \underbrace{\mathbb{E}_{q_\theta(w)}[\mathcal{L}_{\mathcal{D}}(w)]}_{\text{error cost}} + \underbrace{\mathrm{KL}(q_\theta(w) \| p(w))}_{\text{complexity cost}}
$$

To optimize the error cost efficiently, one can use the **Local Reparameterization Trick**. Instead of sampling weights directly, which introduces high variance in gradients, LRT samples the pre-activations.

For a linear layer with inputs $X$, weight means $M$, and variances $V$, the pre-activation $\Gamma$ is distributed as:

$$
\Gamma \sim \mathcal{N}(XM^T, X^2 V^T)
$$

We sample $\zeta = XM^T + \varepsilon \odot \sqrt{X^2 V^T}$, where $\varepsilon \sim \mathcal{N}(0, I)$, allowing for stable, low-variance backpropagation.

---

Alex Graves [*"Practical Variational Inference for Neural Networks"*](https://papers.nips.cc/paper_files/paper/2011/hash/7eb3c8be3d411e8ebfab08eba5f49632-Abstract.html) (2011)

Diederik P. Kingma, Tim Salimans, Max Welling [*"Variational Dropout and the Local Reparameterization Trick"*]() (2015)
