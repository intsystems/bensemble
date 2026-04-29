# Probabilistic Backpropagation

The implementation of PBP (Hernández-Lobato & Adams, 2015) uses Assumed Density Filtering (ADF) in an online fashion.

Means and variances are propagated analytically through the network. For ReLU activations, we use exact moment-matching functions relying on the PDF/CDF of the standard normal distribution.

Weights are updated by matching the moments of the tilted distribution $q(w)p(y|x,w)$. We compute gradients of the log-partition function $\log Z$ to update $\mu$ and $\Sigma$ directly, completely bypassing standard Stochastic Gradient Descent (SGD).

---

José Miguel Hernández-Lobato, Ryan Adams [*"Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks"*](https://proceedings.mlr.press/v37/hernandez-lobatoc15.html) (2015)
