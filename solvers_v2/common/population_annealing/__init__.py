from __future__ import annotations

from collections.abc import Callable

import torch

from solvers_v2.src.observables import Observables, compute_energy
from solvers_v2.src.result import SolverResult
from solvers_v2.src.schedules import schedule_temperatures


UpdateFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
ProgressFn = Callable[[int, int, str], None]


def population_annealing(
    couplings: torch.Tensor,
    pop_size: int,
    num_steps_mc: int,
    t_start: float,
    t_end: float,
    num_temps: int,
    schedule: str,
    update: UpdateFn,
    high_temp_thermalization_steps: int,
    reweight_mode: str = "multinomial",
    device: torch.device | None = None,
    progress_callback: ProgressFn | None = None,
) -> SolverResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_spins = couplings.shape[0]
    couplings = couplings.to(device)
    temperatures = schedule_temperatures(t_start, t_end, num_temps, schedule, num_spins)
    population = (torch.randint(0, 2, (pop_size, num_spins), device=device).float() * 2.0 - 1.0)
    observables = Observables(couplings, num_spins)

    old_temperature = temperatures[0]
    if progress_callback:
        progress_callback(0, num_temps, "thermalization")
    for _ in range(high_temp_thermalization_steps):
        population = update(population, couplings, 1.0 / old_temperature)
    observables.update(population)

    for temp_idx, temperature in enumerate(temperatures[1:], start=1):
        if progress_callback:
            progress_callback(temp_idx, num_temps, f"T={temperature:.3f}")
        delta_beta = 1.0 / temperature - 1.0 / old_temperature
        energies = compute_energy(population, couplings)
        energies = energies - torch.min(energies)
        probabilities = torch.softmax(-energies * delta_beta, dim=0)
        if reweight_mode == "multinomial":
            resampled_indices = torch.multinomial(probabilities, num_samples=pop_size, replacement=True)
        else:
            raise ValueError(f"reweight_mode {reweight_mode!r} not supported")
        population = population[resampled_indices]

        for _ in range(num_steps_mc):
            population = update(population, couplings, 1.0 / temperature)
        observables.update(population)
        old_temperature = temperature

    return SolverResult(temperatures, observables)
