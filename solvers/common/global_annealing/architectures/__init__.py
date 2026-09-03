"""Neural network architectures for global annealing."""

from solvers.common.global_annealing.architectures.base import Architecture
from solvers.common.global_annealing.architectures.made import MADEArchitecture

__all__ = ["Architecture", "MADEArchitecture"]
