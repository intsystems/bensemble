# Uncertainty Analysis

Bensemble allows you to understand why your model is uncertain by decomposing the total predictive variance into two components:

1. **Aleatoric Uncertainty (Data Noise)**:
   Uncertainty inherent in the data (e.g., blurry images). It cannot be reduced by collecting more data.

2. **Epistemic Uncertainty (Model Ignorance)**:
   Uncertainty due to the model's lack of knowledge. This is high for data the model hasn't seen during training (Out-of-Distribution).

### How to compute
Pass your ensemble's predictions to the decomposition functions:

```python
from bensemble.uncertainty import decompose_classification_uncertainty
total, aleatoric, epistemic = decompose_classification_uncertainty(probs)
