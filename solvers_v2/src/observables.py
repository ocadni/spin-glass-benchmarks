from __future__ import annotations

import torch


def compute_energy(spins: torch.Tensor, couplings: torch.Tensor, take_mean: bool = False) -> torch.Tensor:
    energy = -(torch.einsum("ki,ik->k", spins, torch.einsum("ij,kj->ik", couplings, spins))) / 2.0
    if take_mean:
        return energy.mean()
    return energy


class Observables:
    def __init__(self, couplings: torch.Tensor, num_spins: int):
        self.couplings = couplings
        self.num_spins = num_spins
        self.observables: dict[str, list[float]] = {
            "min_energy": [],
            "mean_energy": [],
        }

    def update(self, population: torch.Tensor) -> None:
        energies = compute_energy(population, self.couplings, take_mean=False)
        self.observables["min_energy"].append(float(energies.min() / self.num_spins))
        self.observables["mean_energy"].append(float(energies.mean() / self.num_spins))

    def get_observable_history(self, observable_name: str) -> list[float]:
        if observable_name not in self.observables:
            raise ValueError(f"observable {observable_name!r} not found")
        return self.observables[observable_name]

