from __future__ import annotations

import torch

from solvers_v2.src.observables import Observables
from solvers_v2.src.result import SolverResult
from solvers_v2.src.updates import greedy_update


def greedy(couplings: torch.Tensor, pop_size: int, device: torch.device | None = None) -> SolverResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_spins = couplings.shape[0]
    couplings = couplings.to(device)
    population = (torch.randint(0, 2, (pop_size, num_spins), device=device).float() * 2.0 - 1.0)
    observables = Observables(couplings, num_spins)
    observables.update(population)

    changed = True
    while changed:
        population, changed = greedy_update(population, couplings)

    observables.update(population)
    return SolverResult(None, observables)
