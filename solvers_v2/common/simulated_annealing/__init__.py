from __future__ import annotations

from collections.abc import Callable

import torch

from solvers_v2.src.observables import Observables
from solvers_v2.src.result import SolverResult
from solvers_v2.src.schedules import schedule_temperatures


UpdateFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def simulated_annealing(
    couplings: torch.Tensor,
    pop_size: int,
    num_steps_mc: int,
    t_start: float,
    t_end: float,
    num_temps: int,
    schedule: str,
    update: UpdateFn,
    high_temp_thermalization_steps: int,
    record_final_duplicate: bool,
    device: torch.device | None = None,
) -> SolverResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_spins = couplings.shape[0]
    couplings = couplings.to(device)
    temperatures = schedule_temperatures(t_start, t_end, num_temps, schedule, num_spins)
    population = (torch.randint(0, 2, (pop_size, num_spins), device=device).float() * 2.0 - 1.0)
    observables = Observables(couplings, num_spins)

    for _ in range(high_temp_thermalization_steps):
        population = update(population, couplings, 1.0 / temperatures[0])
    observables.update(population)

    for temperature in temperatures[1:]:
        beta = 1.0 / temperature
        for _ in range(num_steps_mc):
            population = update(population, couplings, beta)
        observables.update(population)

    if record_final_duplicate:
        observables.update(population)
    return SolverResult(temperatures, observables)
