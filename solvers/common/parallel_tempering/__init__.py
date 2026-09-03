from __future__ import annotations

from collections.abc import Callable

import torch

from solvers.src.observables import Observables, compute_energy
from solvers.src.result import SolverResult
from solvers.src.schedules import schedule_temperatures


UpdateFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def parallel_tempering(
    couplings: torch.Tensor,
    num_steps_mc: int,
    swap_interval: int,
    t_start: float,
    t_end: float,
    num_temps: int,
    schedule: str,
    update: UpdateFn,
    high_temp_thermalization_steps: int,
    device: torch.device | None = None,
) -> SolverResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_spins = couplings.shape[0]
    couplings = couplings.to(device)
    temperatures = schedule_temperatures(t_start, t_end, num_temps, schedule, num_spins)
    betas = 1.0 / torch.tensor(temperatures, device=device)
    num_replicas = len(temperatures)
    replicas = (torch.randint(0, 2, (num_replicas, num_spins), device=device).float() * 2.0 - 1.0)
    observables = Observables(couplings, num_spins)

    for replica in range(num_replicas):
        for _ in range(high_temp_thermalization_steps):
            replicas[replica : replica + 1] = update(
                replicas[replica : replica + 1],
                couplings,
                float(betas[replica]),
            )

    observables.update(replicas[-1:])

    for step in range(num_steps_mc):
        for replica in range(num_replicas):
            replicas[replica : replica + 1] = update(
                replicas[replica : replica + 1],
                couplings,
                float(betas[replica]),
            )

        if step % swap_interval == 0:
            for replica in range(num_replicas - 1):
                energy_i = compute_energy(replicas[replica : replica + 1], couplings)[0]
                energy_j = compute_energy(replicas[replica + 1 : replica + 2], couplings)[0]
                delta = (betas[replica + 1] - betas[replica]) * (energy_j - energy_i)
                if torch.log(torch.rand(1)) < delta:
                    temporary = replicas[replica].clone()
                    replicas[replica] = replicas[replica + 1]
                    replicas[replica + 1] = temporary

        observables.update(replicas[-1:])

    return SolverResult(temperatures, observables)
