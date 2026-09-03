"""Global Annealing with ML-enhanced sampling."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from solvers.common.global_annealing.architectures import Architecture, MADEArchitecture
from solvers.src.observables import Observables, compute_energy
from solvers.src.result import SolverResult
from solvers.src.schedules import schedule_temperatures


UpdateFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
ProgressFn = Callable[[int, int, str], None]


def ml_metropolis_update(
    architecture: Architecture,
    model: nn.Module,
    population: torch.Tensor,
    couplings: torch.Tensor,
    beta: float,
    device: torch.device,
    num_steps: int = 1,
) -> torch.Tensor:
    """ML-assisted Metropolis update using neural network model."""
    with torch.no_grad():
        current_config = population.clone()

        for _ in range(num_steps):
            num_configs, num_spins = current_config.shape
            new_config = architecture.generate_configs(model, num_spins, num_configs, device)

            current_energy = compute_energy(current_config, couplings)
            current_probability = architecture.compute_log_probability(model, current_config)

            new_energy = compute_energy(new_config, couplings)
            new_probability = architecture.compute_log_probability(model, new_config)

            arg_new = -beta * new_energy + new_probability
            arg_current = -beta * current_energy + current_probability

            acceptances = (torch.log(torch.rand(size=(num_configs,), device=device)) < (arg_new - arg_current)).int()
            current_config = torch.einsum("i, ij->ij", (1 - acceptances), current_config) + torch.einsum(
                "i, ij->ij", acceptances, new_config
            )

    return current_config


def global_annealing(
    couplings: torch.Tensor,
    pop_size: int,
    num_steps_mc: int,
    swap_step: int,
    t_start: float,
    t_end: float,
    num_temps: int,
    schedule: str,
    update: UpdateFn,
    high_temp_thermalization_steps: int,
    num_epochs_start: int = 40,
    num_epochs_retrain: int = 1,
    batch_size: int = 256,
    architecture: Architecture | None = None,
    device: torch.device | None = None,
    progress_callback: ProgressFn | None = None,
) -> SolverResult:
    """Global annealing with ML-enhanced sampling.

    Combines neural network-based proposal generation with standard MC updates.

    Args:
        couplings: Coupling matrix
        pop_size: Population size
        num_steps_mc: Number of ML-MC steps per temperature
        swap_step: Number of standard MC steps per ML-MC step
        t_start: Starting temperature
        t_end: Ending temperature
        num_temps: Number of temperature steps
        schedule: Temperature schedule type
        update: Standard MC update function
        high_temp_thermalization_steps: Thermalization steps at high T
        num_epochs_start: Initial training epochs
        num_epochs_retrain: Retraining epochs per temperature
        batch_size: Batch size for training
        architecture: Neural network architecture (default: MADEArchitecture)
        device: Device (cuda/cpu)
        progress_callback: Optional progress callback
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if architecture is None:
        architecture = MADEArchitecture()

    num_spins = couplings.shape[0]
    couplings = couplings.to(device)
    temperatures = schedule_temperatures(t_start, t_end, num_temps, schedule, num_spins)
    population = (torch.randint(0, 2, (pop_size, num_spins), device=device).float() * 2.0 - 1.0)
    observables = Observables(couplings, num_spins)

    old_temperature = temperatures[0]
    if progress_callback:
        progress_callback(0, num_temps, "thermalization+training")
    for _ in range(high_temp_thermalization_steps):
        population = update(population, couplings, 1.0 / old_temperature)
    observables.update(population)

    model = architecture.create_model(num_spins)
    model = architecture.train(
        model, population, device, epochs=num_epochs_start, batch_size=batch_size, learning_rate=1e-3
    )

    for temp_idx, temperature in enumerate(temperatures[1:-1], start=1):
        if progress_callback:
            progress_callback(temp_idx, num_temps, f"T={temperature:.3f}")
        beta = 1.0 / temperature
        for _ in range(num_steps_mc):
            population = ml_metropolis_update(architecture, model, population, couplings, beta, device, num_steps=1)
            for _ in range(swap_step):
                population = update(population, couplings, beta)

        model = architecture.retrain(
            model, population, device, epochs=num_epochs_retrain, batch_size=batch_size, learning_rate=1e-3
        )
        observables.update(population)

    temperature = temperatures[-1]
    if progress_callback:
        progress_callback(num_temps - 1, num_temps, f"T={temperature:.3f} (final)")
    beta = 1.0 / temperature
    for _ in range(num_steps_mc):
        population = ml_metropolis_update(architecture, model, population, couplings, beta, device, num_steps=1)
        for _ in range(swap_step):
            population = update(population, couplings, beta)
    observables.update(population)

    return SolverResult(temperatures, observables)
