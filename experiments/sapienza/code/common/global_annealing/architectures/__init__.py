"""Neural network architectures for global annealing."""

from experiments.sapienza.code.common.global_annealing.architectures.base import Architecture
from experiments.sapienza.code.common.global_annealing.architectures.made import MADEArchitecture

__all__ = ["Architecture", "MADEArchitecture"]
