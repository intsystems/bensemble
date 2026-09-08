# Building Ensembles

Bensemble provides a unified way to create ensembles from various sources of diversity. Whether you have multiple independent models or a single stochastic model (like MC-Dropout), you wrap them in the `Ensemble` class to use the evaluation toolkit.

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

`Ensemble.from_stochastic` switches dropout on for you. To do the same by hand, for instance inside a custom evaluation loop, use `enable_dropout`: it puts every `nn.Dropout` back into training mode while the rest of the model stays in eval mode.

```python
import torch
import torch.nn as nn
from bensemble.utils import enable_dropout

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1))
model.eval()
enable_dropout(model)

x = torch.randn(8, 10)
with torch.no_grad():
    samples = torch.stack([model(x) for _ in range(30)])  # [30, 8, 1]
mean, std = samples.mean(0), samples.std(0)
```

### Variational Inference (Bayesian Layers)

If you built your model using `BayesianLinear` or `BayesianConv2d`, it uses weight sampling to represent uncertainty. Both layers are drop-in replacements for their `torch.nn` counterparts, so a Bayesian CNN looks like an ordinary one:

```python
import torch
import torch.nn as nn
from bensemble.core.ensemble import Ensemble
from bensemble.layers import BayesianConv2d, BayesianLinear
from bensemble.losses import VariationalLoss
from bensemble.utils import get_total_kl

model = nn.Sequential(
    BayesianConv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    BayesianLinear(8 * 14 * 14, 10),
)

# For classification the likelihood is a per-sample cross-entropy
criterion = VariationalLoss(nn.CrossEntropyLoss(reduction="none"), num_batches=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

images, labels = torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))
model.train()
for _ in range(50):  # Dummy loop
    optimizer.zero_grad()
    loss = criterion(model(images), labels, get_total_kl(model))
    loss.backward()
    optimizer.step()

# Wrap as an ensemble
ensemble = Ensemble.from_stochastic(model, num_samples=30, mode="bayesian")
```

Once trained, weights whose posterior mean is small relative to its standard deviation carry little information. `prune_model` zeroes them across every Bayesian layer and reports the fraction removed:

```python
from bensemble.utils import prune_model

sparsity = prune_model(model, threshold=0.83)
print(f"pruned {sparsity:.0%} of Bayesian weights")
```

The threshold is a signal-to-noise ratio, so the pruned fraction depends on how well the model is trained: a barely trained network loses most of its weights, a converged one far fewer.

---

## 3. Automated Ensemble Search (NAS)

Bensemble also provides **Neural Ensemble Search (NES)**. Instead of manually picking architectures, these algorithms find the best combination for you.

### Evolutionary Search (NES-RE)

```python
from bensemble.search.nes import EvolutionarySearcher

searcher = EvolutionarySearcher(search_space, train_fn=my_trainer, pool_size=50, ensemble_size=5)
# Returns a ready-to-use Ensemble object
ensemble = searcher.search(val_loader)
```

### Bayesian Sampling (NESBS)

```python
from bensemble.search.bayesian import NESBayesianSampler

sampler = NESBayesianSampler(search_space, train_fn=my_trainer, pool_size=50, ensemble_size=5)
# Samples a diverse ensemble from the candidate pool using SVGD
ensemble = sampler.sample_svgd(val_loader)
```

### Selecting members from a pool

The searchers above train their own candidates. If you already have a pool of trained models, `forward_select` picks the subset that works best together: it adds members greedily, one at a time, keeping whichever candidate most improves the criterion on a validation set.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from bensemble.core.ensemble import Ensemble
from bensemble.search.selection import forward_select, classification_nll_criterion

pool = [model1, model2, model3, model4, model5, model6, model7, model8]
val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)

members = forward_select(
    pool,
    val_loader,
    ensemble_size=3,
    device=torch.device("cpu"),
    criterion=classification_nll_criterion,
)
ensemble = Ensemble.from_models(members)
```

`classification_nll_criterion` scores a candidate set by the negative log-likelihood of its averaged softmax; `regression_mse_criterion` does the same with mean squared error. Any callable with the signature `(members, val_loader, device) -> float` works, lower being better.

---

## The Unified Output

Regardless of how you created the `ensemble` object, you now have access to the standardized `predict_members` method:

```python
# All ensembles return a tensor of shape [M_models, Batch_size, Output_dim]
member_outputs = ensemble.predict_members(x_test)
```

This output can be passed directly to our [Uncertainty Analysis](uncertainty-analysis.md) and [Metrics](../api/metrics.md) modules.
