# Laplace Approximation

A scalable Laplace approximation using Kronecker-Factored Approximate Curvature.

Since this is a post-hoc method, a standard deterministic network is first trained to find the MAP estimate $W^{\text{MAP}}_l$. We then capture the covariance of activations ($A_l$) and pre-activation gradients ($G_l$).

The posterior for layer $l$ is then approximated as a matrix normal distribution $\mathcal{MN}(W^{\text{MAP}}_l, A_l, G_l)$. Samples are generated efficiently using the Cholesky decomposition of Kronecker factors:

$$
W_{\text{sample}} = W_{\text{MAP}} + L_V Z L_U^T
$$

where $L_V, L_U$ are Cholesky factors of the inverse regularized covariances and $Z$ is sampled from the standard matrix normal distribution.

---

Hippolyt Ritter, Aleksandar Botev, David Barber[*"A Scalable Laplace Approximation for Neural Networks"*](https://discovery.ucl.ac.uk/id/eprint/10080902/1/kflaplace.pdf) (2018)
