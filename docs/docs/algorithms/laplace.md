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

The posterior for layer $l$ is then approximated as a matrix normal distribution $\mathcal{MN}(W^{\text{MAP}}_l, A_l, G_l)$. Samples are generated from the two Kronecker factors independently:

$$
W_{\text{sample}} = W_{\text{MAP}} + L_V Z L_U^T
$$

where $L_V, L_U$ are symmetric square roots of the inverse regularized covariances, computed by eigendecomposition with the eigenvalues clamped from below for stability, and $Z$ is sampled from the standard matrix normal distribution.

---

Hippolyt Ritter, Aleksandar Botev, David Barber[*"A Scalable Laplace Approximation for Neural Networks"*](https://discovery.ucl.ac.uk/id/eprint/10080902/1/kflaplace.pdf) (2018)
