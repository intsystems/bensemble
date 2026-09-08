# Uncertainty Analysis

Bensemble allows you to understand why your model is uncertain by decomposing the total predictive variance into two components:

1. **Aleatoric Uncertainty (Data Noise)**:
   Uncertainty inherent in the data (e.g., blurry images). It cannot be reduced by collecting more data.

2. **Epistemic Uncertainty (Model Ignorance)**:
   Uncertainty due to the model's lack of knowledge. This is high for data the model hasn't seen during training (Out-of-Distribution).

### How to compute
Pass your ensemble's predictions to the decomposition functions.

For classification, `probs` holds one softmax per member, shaped `[M_models, Batch_size, Num_classes]`:

```python
from bensemble.uncertainty import decompose_classification_uncertainty

total, aleatoric, epistemic = decompose_classification_uncertainty(probs)
```

With $M$ members and $\bar p = \frac{1}{M}\sum_m p_m$ the averaged prediction, total uncertainty is the entropy of the average, aleatoric is the average entropy, and epistemic is their difference, the mutual information between the prediction and the model:

$$
\underbrace{\mathcal{H}[\bar p]}_{\text{total}}
= \underbrace{\frac{1}{M}\sum_{m=1}^{M} \mathcal{H}[p_m]}_{\text{aleatoric}}
+ \underbrace{\mathcal{H}[\bar p] - \frac{1}{M}\sum_{m=1}^{M} \mathcal{H}[p_m]}_{\text{epistemic}}
$$

For regression, each member has to predict a variance alongside its mean. A common way is a network with two outputs, one read as the mean and the other passed through `softplus` to stay positive:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from bensemble.core.ensemble import Ensemble
from bensemble.uncertainty import decompose_regression_uncertainty

regressors = [nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2)) for _ in range(5)]
ensemble = Ensemble.from_models(regressors)

x = torch.randn(16, 4)
out = ensemble.predict_members(x)  # [5, 16, 2]
means, variances = out[..., :1], F.softplus(out[..., 1:])

total, aleatoric, epistemic = decompose_regression_uncertainty(means, variances)
```

With member means $\mu_m$ and variances $\sigma_m^2$, aleatoric uncertainty is the average predicted variance, epistemic is the spread of the means, and the two add up to the total by the law of total variance:

$$
\sigma^2_{\text{total}}
= \underbrace{\frac{1}{M}\sum_{m=1}^{M} \sigma_m^2}_{\text{aleatoric}}
+ \underbrace{\frac{1}{M}\sum_{m=1}^{M} \bigl(\mu_m - \bar\mu\bigr)^2}_{\text{epistemic}},
\qquad \bar\mu = \frac{1}{M}\sum_{m=1}^{M} \mu_m
$$
