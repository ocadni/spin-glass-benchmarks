from __future__ import annotations

from pathlib import Path

from solvers_v2 import common
from solvers_v2.src.io import load_pairwise_couplings
from solvers_v2.src.reproducibility import set_reproducible_seed
from solvers_v2.src.result import SolverResult
from solvers_v2.src.updates import (
    checkerboard_indices_2d,
    checkerboard_indices_3d,
    checkerboard_metropolis_update,
)


def run_baseline_case(
    algorithm: str,
    fixture: str | Path,
    parameters: dict,
    seed: int,
    dimension: str,
) -> SolverResult:
    set_reproducible_seed(seed)
    couplings = load_pairwise_couplings(fixture, symmetric=True)
    if dimension == "2d":
        index_sets = checkerboard_indices_2d(parameters["L"])
    elif dimension == "3d":
        index_sets = checkerboard_indices_3d(parameters["L"])
    else:
        raise ValueError(f"unsupported EA dimension: {dimension}")

    def update(population, update_couplings, beta):
        return checkerboard_metropolis_update(population, update_couplings, beta, index_sets)

    if algorithm == "simulated_annealing":
        return common.simulated_annealing(
            couplings,
            pop_size=parameters["pop_size"],
            num_steps_mc=parameters["MCsteps"],
            t_start=parameters["Tstart"],
            t_end=parameters["Tend"],
            num_temps=parameters["num_temps"],
            schedule=parameters["schedule"],
            update=update,
            high_temp_thermalization_steps=parameters["high_temp_thermalization_steps"],
            record_final_duplicate=False,
        )
    if algorithm == "population_annealing":
        return common.population_annealing(
            couplings,
            pop_size=parameters["pop_size"],
            num_steps_mc=parameters["MCsteps"],
            t_start=parameters["Tstart"],
            t_end=parameters["Tend"],
            num_temps=parameters["num_temps"],
            schedule=parameters["schedule"],
            update=update,
            high_temp_thermalization_steps=parameters["high_temp_thermalization_steps"],
        )
    raise ValueError(f"unsupported EA algorithm: {algorithm}")
