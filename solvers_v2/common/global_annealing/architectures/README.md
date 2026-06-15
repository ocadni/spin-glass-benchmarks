# Global Annealing Architectures

This directory contains neural network architectures that can be used with the global annealing algorithm.

## Overview

The global annealing algorithm uses machine learning-enhanced sampling where a neural network learns to propose new spin configurations. Different neural network architectures can be plugged in via the `Architecture` base class.

## Available Architectures

### MADE (Default)

**MADE** (Masked Autoencoder for Distribution Estimation) is the default architecture. It uses autoregressive masking to model the spin configuration distribution.

**Features:**
- Single linear layer with autoregressive masking
- Fast sampling (optimized forward_n method)
- Training with early stopping based on validation loss
- Retraining support for online learning during annealing

**Reference:** Germain et al., "MADE: Masked Autoencoder for Distribution Estimation", ICML 2015

## Usage

### Default (MADE)

```python
from solvers_v2 import common

result = common.global_annealing(
    couplings=J,
    pop_size=1000,
    num_steps_mc=10,
    swap_step=5,
    t_start=10.0,
    t_end=0.1,
    num_temps=50,
    schedule="exponential",
    update=sequential_metropolis_update,
    high_temp_thermalization_steps=100,
    # architecture defaults to MADEArchitecture()
)
```

### Specifying an Architecture

```python
from solvers_v2 import common

# Option 1: Pass architecture string in parameters
parameters = {
    "pop_size": 1000,
    "MLMCsteps": 10,
    "swap_step": 5,
    "Tstart": 10.0,
    "Tend": 0.1,
    "num_temps": 50,
    "schedule": "exponential",
    "high_temp_thermalization_steps": 100,
    "architecture": "made",  # Specify architecture by name
}

# Option 2: Pass architecture instance directly to global_annealing
architecture = common.MADEArchitecture()
result = common.global_annealing(
    couplings=J,
    architecture=architecture,
    # ... other parameters
)
```

## Creating a Custom Architecture

To add a new architecture, create a new Python file in this directory and subclass `Architecture`:

```python
from solvers_v2.common.global_annealing.architectures.base import Architecture
import torch
import torch.nn as nn

class MyArchitecture(Architecture):
    def create_model(self, input_size: int) -> nn.Module:
        # Return a new model instance
        pass
    
    def train(self, model, dataset, device, epochs, batch_size, learning_rate, **kwargs):
        # Train the model
        pass
    
    def retrain(self, model, dataset, device, epochs, batch_size, learning_rate, **kwargs):
        # Retrain the model
        pass
    
    def generate_configs(self, model, num_spins, num_configs, device):
        # Generate spin configurations
        pass
    
    def compute_log_probability(self, model, configs):
        # Compute log probability of configurations
        pass
```

Then register it in `__init__.py` and add support in the solver adapters (`solvers_v2/families/*/`).

## Architecture Interface

All architectures must implement the `Architecture` abstract base class with the following methods:

- `create_model(input_size)`: Create a new model instance
- `train(model, dataset, device, epochs, batch_size, learning_rate, **kwargs)`: Initial training
- `retrain(model, dataset, device, epochs, batch_size, learning_rate, **kwargs)`: Online retraining
- `generate_configs(model, num_spins, num_configs, device)`: Generate spin configurations
- `compute_log_probability(model, configs)`: Compute log probabilities for Metropolis acceptance

See [base.py](base.py) for the full interface specification.
