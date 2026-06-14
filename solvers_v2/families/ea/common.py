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
    progress_callback=None,
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
            progress_callback=progress_callback,
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
            progress_callback=progress_callback,
        )
    if algorithm == "global_annealing":
        return common.global_annealing(
            couplings,
            pop_size=parameters["pop_size"],
            num_steps_mc=parameters["MLMCsteps"],
            swap_step=parameters["swap_step"],
            t_start=parameters["Tstart"],
            t_end=parameters["Tend"],
            num_temps=parameters["num_temps"],
            schedule=parameters["schedule"],
            update=update,
            high_temp_thermalization_steps=parameters["high_temp_thermalization_steps"],
            num_epochs_start=parameters.get("num_epochs_start", 40),
            num_epochs_retrain=parameters.get("num_epochs_retrain", 1),
            batch_size=parameters.get("batch_size", 256),
            progress_callback=progress_callback,
        )
    raise ValueError(f"unsupported EA algorithm: {algorithm}")
