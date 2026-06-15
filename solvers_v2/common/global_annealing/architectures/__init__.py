"""Neural network architectures for global annealing."""

from solvers_v2.common.global_annealing.architectures.base import Architecture
from solvers_v2.common.global_annealing.architectures.made import MADEArchitecture

__all__ = ["Architecture", "MADEArchitecture"]
