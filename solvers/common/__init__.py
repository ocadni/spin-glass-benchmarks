"""Common algorithms shared across all spin glass families."""

from solvers.common.global_annealing import global_annealing
from solvers.common.global_annealing.architectures import Architecture, MADEArchitecture
from solvers.common.greedy import greedy
from solvers.common.parallel_tempering import parallel_tempering
from solvers.common.population_annealing import population_annealing
from solvers.common.simulated_annealing import simulated_annealing

__all__ = [
    "simulated_annealing",
    "population_annealing",
    "parallel_tempering",
    "greedy",
    "global_annealing",
    "Architecture",
    "MADEArchitecture",
]
