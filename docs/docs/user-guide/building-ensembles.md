# Building Ensembles

Bensemble provides a unified way to create ensembles from various sources of diversity. Whether you have multiple independent models or a single stochastic model (like MC-Dropout), you wrap them in the `Ensemble` class to use our evaluation toolkit.

There are three primary ways to build an ensemble.

---

## 1. Explicit Ensembles (Deep Ensembles)

The simplest way is to combine a list of independent, pre-trained PyTorch models. This is often referred to as a **Deep Ensemble**.

```python
import torch.nn as nn
from bensemble.core.ensemble import Ensemble

# Assume you have trained 5 different models
models = [model1, model2, model3, model4, model5]

# Create an ensemble
ensemble = Ensemble.from_models(models)

print(f"Ensemble size: {ensemble.num_members}")
```

Use this method when you have the computational budget to train and store multiple distinct networks.

---

## 2. Implicit Ensembles (Stochastic Models)

Implicit ensembles use a single model that produces different outputs for the same input due to internal randomness.

### MC Dropout

If your model contains `nn.Dropout` layers, you can treat it as an ensemble by keeping dropout active during inference.

```python
from bensemble.core.ensemble import Ensemble

# A standard model with nn.Dropout layers
model = MyDropoutModel() 

# Wrap as an ensemble with 30 stochastic forward passes
ensemble = Ensemble.from_stochastic(model, num_samples=30, mode="dropout")
```

### Variational Inference (Bayesian Layers)

If you built your model using `BayesianLinear` or `BayesianConv2d`, it uses weight sampling to represent uncertainty.

```python
from bensemble.core.ensemble import Ensemble

# A model built with bensemble.layers
bayesian_model = MyBayesianModel() 

# Wrap as an ensemble
ensemble = Ensemble.from_stochastic(bayesian_model, num_samples=30, mode="bayesian")
```

---

## 3. Automated Ensemble Search (NAS)

Bensemble also provides **Neural Ensemble Search (NES)**. Instead of manually picking architectures, these algorithms find the best combination for you.

### Evolutionary Search (NES-RE)

```python
from bensemble.search.nes import EvolutionarySearcher

searcher = EvolutionarySearcher(search_space, pool_size=50)
# Returns a ready-to-use Ensemble object
ensemble = searcher.search(train_fn=my_trainer, val_loader=val_loader)
```

### Bayesian Sampling (NESBS)

```python
from bensemble.search.bayesian import NESBayesianSampler

sampler = NESBayesianSampler(trained_supernet)
# Samples an ensemble from the supernet using SVGD
ensemble = sampler.sample_svgd(val_loader, n_models=5)
```

---

## The Unified Output

Regardless of how you created the `ensemble` object, you now have access to the standardized `predict_members` method:

```python
# All ensembles return a tensor of shape [M_models, Batch_size, Output_dim]
member_outputs = ensemble.predict_members(x_test)
```

This output can be passed directly to our [Uncertainty Analysis](uncertainty-analysis.md) and [Metrics](metrics.md) modules.
