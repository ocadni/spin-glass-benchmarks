"""Common algorithms shared across all spin glass families."""

from solvers_v2.common.global_annealing import global_annealing
from solvers_v2.common.greedy import greedy
from solvers_v2.common.parallel_tempering import parallel_tempering
from solvers_v2.common.population_annealing import population_annealing
from solvers_v2.common.simulated_annealing import simulated_annealing

__all__ = [
    "simulated_annealing",
    "population_annealing",
    "parallel_tempering",
    "greedy",
    "global_annealing",
]
