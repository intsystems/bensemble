# Calibration & Metrics

A classifier is calibrated when its confidence matches its accuracy: among the predictions it makes with 80% confidence, about 80% should be right. Ensembles usually come out better calibrated than single networks, but rarely perfectly so, and post-hoc scaling closes the rest of the gap.

Everything on this page is for classification: it expects class probabilities or logits and integer labels.

## Post-hoc scaling

Both scalers learn a transformation of the logits $z$ on a hold-out validation set, then apply it at test time. `TemperatureScaling` divides all logits by one scalar, which keeps the ranking of classes and therefore the accuracy. `VectorScaling` learns a scale and a shift per class, so it can also fix class-specific over- or under-confidence, at the cost of being able to change predictions.

$$
p = \operatorname{softmax}(z / T)
\qquad\qquad
p = \operatorname{softmax}(a \odot z + b)
$$

Both are fitted by minimizing the negative log-likelihood on the validation set.

```python
import torch
from bensemble.calibration.scaling import VectorScaling

# logits and labels from a validation set the model was not trained on
scaler = VectorScaling(num_classes=3).fit(val_logits, val_labels)

# apply to new logits before the softmax
probs = torch.softmax(scaler(test_logits), dim=-1)
```

For an ensemble, fit the scaler on the logits you actually evaluate, whether that is the per-member output of `predict_members` or its mean.

## Scoring rules

`negative_log_likelihood` and `brier_score` are proper scoring rules: the best possible score is reached only by the true probabilities, so they reward calibration and accuracy at the same time. For $N$ samples with predicted probabilities $p_{ik}$ over $K$ classes and true labels $y_i$:

$$
\mathrm{NLL} = -\frac{1}{N}\sum_{i=1}^{N} \log p_{i,y_i}
\qquad\qquad
\mathrm{Brier} = \frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}\bigl(p_{ik} - \mathbb{1}[y_i = k]\bigr)^2
$$

The Brier score sums over all classes, so on a binary task it is twice the value of the single-probability form found in some references.

`expected_calibration_error` isolates calibration alone. It takes each prediction's confidence $\max_k p_{ik}$, splits the $[0, 1]$ range into equal-width bins $B_1, \dots, B_n$ (15 by default), and compares the average confidence in each bin with the accuracy in it, weighted by how many predictions the bin holds:

$$
\mathrm{ECE} = \sum_{b=1}^{n} \frac{|B_b|}{N}\,\bigl|\operatorname{acc}(B_b) - \operatorname{conf}(B_b)\bigr|
$$

```python
from bensemble.metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)

for name, logits in [("before", test_logits), ("after", scaler(test_logits))]:
    p = torch.softmax(logits, dim=-1)
    print(
        f"{name}: NLL={negative_log_likelihood(p, test_labels):.3f} "
        f"Brier={brier_score(p, test_labels):.3f} "
        f"ECE={expected_calibration_error(p, test_labels):.3f}"
    )
```

Lower is better for all three. Report NLL or Brier when you care about the predictive distribution as a whole, and ECE when the question is specifically whether the confidences can be trusted.
