# Laplace Approximation

A scalable Laplace approximation using Kronecker-Factored Approximate Curvature.

Since this is a post-hoc method, a standard deterministic network is first trained to find the MAP estimate $W^{\text{MAP}}_l$. We then estimate input second moments ($A_l$) and pre-activation curvature ($G_l$) for each linear layer.

The pre-activation curvature is a generalized Gauss–Newton approximation:
$G_l = \mathbb{E}[J_l^T H_{\text{out}} J_l]$, where $J_l$ is the Jacobian of the
model output with respect to the layer's pre-activations. Autograd computes these
Jacobians through the model's activations and branches. This includes zero
curvature through an inactive ReLU; for smooth activations, it is not the full
loss Hessian.

Curvature estimation supports `nn.Linear` layers with 2D inputs, each called once
per forward pass. Examples must be independent in eval mode. It uses one backward
pass per output coordinate and leaves parameter gradients unchanged. Only Linear
weights are sampled; all biases and other parameters remain fixed.

By default (`regularization="exact"`), the precision of the layer's weights,
flattened row by row, is

$$
P_l = N (G_l \otimes A_l) + \tau I,
$$

where $N$ is the training dataset size and $\tau$ is `prior_precision`.
For numerical stability, `damping` is added to the diagonal of $P_l$ before sampling.
`num_samples` limits how many examples are used to estimate the per-example
averages $A_l$ and $G_l$. In their joint eigenbasis, each weight perturbation
has variance $1/(N\,g_i a_j + \tau)$ at unit temperature and zero damping.
This adds the prior exactly within the K-FAC approximation.

Use `regularization="legacy"` for the old scaling, which multiplies each
factor by $N$. The `"paper"` mode regularizes the two factors separately
using $\sqrt{N}$ and $\sqrt{\tau}$.

---

Hippolyt Ritter, Aleksandar Botev, David Barber[*"A Scalable Laplace Approximation for Neural Networks"*](https://discovery.ucl.ac.uk/id/eprint/10080902/1/kflaplace.pdf) (2018)
