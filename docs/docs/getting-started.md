# Getting Started

To get started with Bensemble, all you'll need is Python with at least version 3.10.

## Installation

```
pip install bensemble
```

## Basic usage

All Bensemble classes are built as wrappers around PyTorch models that are subclasses of the `torch.nn.Module` class. All you need then is a model, a `torch.utils.data.DataLoader` for training and you're good to go!

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from bensemble.methods.variational_inference import VariationalEnsemble

# Create a model
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Choose your data
train_data = ...
test_data = ...


# Create a DataLoader instance
train_loader = DataLoader(data)

# Create ensemble
ensemble = VariationalEnsemble(model)

# Train model and its posterior
ensemble.fit(train_loader)

# Make predictions
for x in test_data:
    print(ensemble.predict(x, n_samples=10))
```