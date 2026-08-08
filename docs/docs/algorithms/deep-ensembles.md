# Deep Ensembles

A Deep Ensemble consists of several independently-trained neural networks, each initialized with different random weights and typically trained on the same (or bootstrapped) data. At inference time, each member produces its own prediction, and these predictions are combined — for classification, by averaging the predicted class probabilities.

Because each member converges to a different point in the loss landscape, disagreement between members' predictions provides a simple and effective estimate of epistemic uncertainty, without requiring any changes to how the individual models are trained.

---

Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell [*"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"*](https://arxiv.org/abs/1612.01474) (2017)