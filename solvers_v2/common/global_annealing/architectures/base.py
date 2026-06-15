"""Base architecture interface for global annealing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Architecture(ABC):
    """Base class for neural network architectures used in global annealing."""

    @abstractmethod
    def create_model(self, input_size: int) -> nn.Module:
        """Create and return a new model instance.

        Args:
            input_size: Number of spins in the system

        Returns:
            PyTorch model instance
        """
        pass

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        dataset: torch.Tensor,
        device: torch.device,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        **kwargs,
    ) -> nn.Module:
        """Train the model on a dataset.

        Args:
            model: Model to train
            dataset: Training data (spin configurations)
            device: Device to train on
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            **kwargs: Additional architecture-specific parameters

        Returns:
            Trained model
        """
        pass

    @abstractmethod
    def retrain(
        self,
        model: nn.Module,
        dataset: torch.Tensor,
        device: torch.device,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        **kwargs,
    ) -> nn.Module:
        """Retrain an existing model on new data.

        Args:
            model: Model to retrain
            dataset: Training data (spin configurations)
            device: Device to train on
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            **kwargs: Additional architecture-specific parameters

        Returns:
            Retrained model
        """
        pass

    @abstractmethod
    def generate_configs(
        self,
        model: nn.Module,
        num_spins: int,
        num_configs: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate spin configurations using the model.

        Args:
            model: Trained model
            num_spins: Number of spins
            num_configs: Number of configurations to generate
            device: Device to generate on

        Returns:
            Generated spin configurations (values in {-1, +1})
        """
        pass

    @abstractmethod
    def compute_log_probability(
        self,
        model: nn.Module,
        configs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of configurations under the model.

        Args:
            model: Trained model
            configs: Spin configurations

        Returns:
            Log probabilities for each configuration
        """
        pass
