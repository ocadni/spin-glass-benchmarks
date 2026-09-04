"""Neural network architectures for global annealing."""

from experiments.sapienza.common.global_annealing.architectures.base import Architecture
from experiments.sapienza.common.global_annealing.architectures.made import MADEArchitecture

__all__ = ["Architecture", "MADEArchitecture"]
