# Monte-Carlo Dropout

For a neural network trained with dropout layers, this method works by leaving dropout active during the inference stage. By performing multiple forward passes with different dropout masks, one obtain an ensemble of models, each with a random weight subsample from the original model.

This approach provides a computationally cheap way to approximate Bayesian inference without modifying the underlying model architecture.

---

Yarin Gal, Zoubin Ghahramani [*"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"*](https://arxiv.org/abs/1506.02142) (2016)
