# Model Calibration

Deep Neural Networks are often **overconfident**. A model might predict a class with 99% probability but only be right 70% of the time. 

### Temperature Scaling

We provide `TemperatureScaling`, a simple yet effective post-hoc method to fix this. It divides the logits by a learned scalar $T$, softening the probabilities without changing the model's accuracy.

### Expected Calibration Error (ECE)

To measure how well your model is calibrated, use the `expected_calibration_error` metric. A perfectly calibrated model has an ECE of 0.
