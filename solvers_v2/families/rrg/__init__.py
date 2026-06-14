"""Random-regular graph solver adapters."""
from __future__ import annotations

from pathlib import Path

from solvers_v2 import common
from solvers_v2.src.io import load_pairwise_couplings
from solvers_v2.src.reproducibility import set_reproducible_seed
from solvers_v2.src.updates import sequential_metropolis_update


def run_pairwise_case(algorithm: str, fixture: str | Path, parameters: dict, seed: int):
    set_reproducible_seed(seed)
    couplings = load_pairwise_couplings(fixture, symmetric=True)

    if algorithm == "simulated_annealing":
        return common.simulated_annealing(
            couplings,
            pop_size=parameters["pop_size"],
            num_steps_mc=parameters["MCsteps"],
            t_start=parameters["Tstart"],
            t_end=parameters["Tend"],
            num_temps=parameters["num_temps"],
            schedule=parameters["schedule"],
            update=sequential_metropolis_update,
            high_temp_thermalization_steps=parameters["high_temp_thermalization_steps"],
            record_final_duplicate=True,
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
            update=sequential_metropolis_update,
            high_temp_thermalization_steps=parameters["high_temp_thermalization_steps"],
        )
    if algorithm == "parallel_tempering":
        return common.parallel_tempering(
            couplings,
            num_steps_mc=parameters["MCsteps"],
            swap_interval=parameters["swap_interval"],
            t_start=parameters["Tstart"],
            t_end=parameters["Tend"],
            num_temps=parameters["num_temps"],
            schedule=parameters["schedule"],
            update=sequential_metropolis_update,
            high_temp_thermalization_steps=parameters["high_temp_thermalization_steps"],
        )
    if algorithm == "greedy":
        return common.greedy(
            couplings,
            pop_size=parameters["pop_size"],
        )
    raise ValueError(f"unsupported RRG algorithm: {algorithm}")

