from __future__ import annotations

from collections.abc import Callable

import torch

from experiments.sapienza.code.src.device import default_device
from experiments.sapienza.code.src.observables import Observables
from experiments.sapienza.code.src.result import SolverResult
from experiments.sapienza.code.src.schedules import schedule_temperatures


UpdateFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
ProgressFn = Callable[[int, int, str], None]


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
    progress_callback: ProgressFn | None = None,
) -> SolverResult:
    if device is None:
        device = default_device()

    num_spins = couplings.shape[0]
    couplings = couplings.to(device)
    temperatures = schedule_temperatures(t_start, t_end, num_temps, schedule, num_spins)
    population = (torch.randint(0, 2, (pop_size, num_spins), device=device).float() * 2.0 - 1.0)
    observables = Observables(couplings, num_spins)

    if progress_callback:
        progress_callback(0, num_temps, "thermalization")
    for _ in range(high_temp_thermalization_steps):
        population = update(population, couplings, 1.0 / temperatures[0])
    observables.update(population)

    for temp_idx, temperature in enumerate(temperatures[1:], start=1):
        if progress_callback:
            progress_callback(temp_idx, num_temps, f"T={temperature:.3f}")
        beta = 1.0 / temperature
        for _ in range(num_steps_mc):
            population = update(population, couplings, beta)
        observables.update(population)

    if record_final_duplicate:
        observables.update(population)
    return SolverResult(temperatures, observables)
