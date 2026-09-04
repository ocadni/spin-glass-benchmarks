"""MADE (Masked Autoencoder for Distribution Estimation) architecture."""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from experiments.sapienza.common.global_annealing.architectures.base import Architecture


class AutoregressiveMasking:
    """Autoregressive constraint for weight matrices."""

    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module: nn.Module) -> None:
        if hasattr(module, "weight"):
            w = module.weight.data
            w = torch.tril(w, -1)
            module.weight.data = w


class MADE(nn.Module):
    """Autoregressive MADE model for spin configuration distribution."""

    def __init__(self, input_size: int):
        super().__init__()
        self.layer = nn.Linear(input_size, input_size, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = self.activation(2 * x)
        return x

    def forward_n(self, input: torch.Tensor, n: int) -> torch.Tensor:
        """Forward pass for the n-th spin only (optimized for generation)."""
        nth_row = self.layer.weight[n]
        x = torch.einsum("ij, j->i", input[:, :n], nth_row[:n])
        x = self.activation(2 * x)
        return x


class MADEArchitecture(Architecture):
    """MADE architecture implementation for global annealing."""

    def create_model(self, input_size: int) -> nn.Module:
        """Create a new MADE model."""
        return MADE(input_size)

    def train(
        self,
        model: nn.Module,
        dataset: torch.Tensor,
        device: torch.device,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        patience: int = 10,
        scheduler_time: int = 10,
        **kwargs,
    ) -> nn.Module:
        """Train a MADE model on spin configurations."""
        data = dataset.clone()
        model = model.to(device)
        model.train()
        clipper = AutoregressiveMasking()
        model.apply(clipper)

        best_loss = float("inf")
        epochs_since_best = 0
        best_weights = None

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss(reduction="sum")
        scheduler = ExponentialLR(optimizer, gamma=0.5)

        for epoch in range(epochs):
            tot_loss = 0.0
            count = 0
            for i in range(0, len(data), batch_size):
                indices = random.sample(range(data.shape[0]), min(batch_size, len(data)))
                batch_data = data[indices].to(device)

                output = model(batch_data)
                loss = criterion(output, (batch_data + 1) / 2)
                tot_loss += loss.item()
                count += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.apply(clipper)

            if epoch % scheduler_time == 1 and epoch > 0:
                scheduler.step()

            avg_loss = tot_loss / count
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_since_best = 0
                best_weights = model.state_dict()
            else:
                epochs_since_best += 1

            if epochs_since_best >= patience:
                break

        if best_weights is not None:
            model.load_state_dict(best_weights)
        return model

    def retrain(
        self,
        model: nn.Module,
        dataset: torch.Tensor,
        device: torch.device,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        **kwargs,
    ) -> nn.Module:
        """Retrain an existing MADE model on new data."""
        data = dataset.clone()
        model.train()
        clipper = AutoregressiveMasking()
        model.apply(clipper)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss(reduction="sum")

        for _ in range(epochs):
            for i in range(0, len(data), batch_size):
                indices = random.sample(range(data.shape[0]), min(batch_size, len(data)))
                batch_data = data[indices].to(device)

                output = model(batch_data)
                loss = criterion(output, (batch_data + 1) / 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.apply(clipper)

        return model

    def generate_configs(
        self,
        model: nn.Module,
        num_spins: int,
        num_configs: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate spin configurations using MADE model autoregressively."""
        with torch.no_grad():
            config = torch.zeros((num_configs, num_spins), device=device)
            for n in range(num_spins):
                probs = model.forward_n(config, n)
                config[:, n] = (torch.bernoulli(probs) * 2 - 1)
        return config

    def compute_log_probability(
        self,
        model: nn.Module,
        configs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of configurations under MADE model."""
        bce = nn.BCELoss(reduction="none")
        with torch.no_grad():
            return -torch.sum(bce(model(configs), (configs + 1) / 2), dim=1)
