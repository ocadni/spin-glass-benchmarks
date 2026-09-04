"""Common algorithms shared across all spin glass families."""

from experiments.sapienza.code.common.global_annealing import global_annealing
from experiments.sapienza.code.common.global_annealing.architectures import Architecture, MADEArchitecture
from experiments.sapienza.code.common.greedy import greedy
from experiments.sapienza.code.common.parallel_tempering import parallel_tempering
from experiments.sapienza.code.common.population_annealing import population_annealing
from experiments.sapienza.code.common.simulated_annealing import simulated_annealing

__all__ = [
    "simulated_annealing",
    "population_annealing",
    "parallel_tempering",
    "greedy",
    "global_annealing",
    "Architecture",
    "MADEArchitecture",
]
