from __future__ import annotations

import torch


def sequential_metropolis_update(population: torch.Tensor, couplings: torch.Tensor, beta: float) -> torch.Tensor:
    updated = population.clone()
    pop_size, num_spins = updated.shape
    for index in range(num_spins):
        proposed = updated.clone()
        proposed[:, index] *= -1
        local_field = torch.einsum("kj,j->k", updated, couplings[index, :])
        delta_energy = -2.0 * proposed[:, index] * local_field
        acceptance_prob = torch.exp(-beta * delta_energy)
        random_values = torch.rand(pop_size, device=updated.device)
        accept = (delta_energy < 0) | (random_values < acceptance_prob)
        updated[:, index] = torch.where(accept, proposed[:, index], updated[:, index])
    return updated


def checkerboard_metropolis_update(
    population: torch.Tensor,
    couplings: torch.Tensor,
    beta: float,
    index_sets: list[torch.Tensor],
) -> torch.Tensor:
    updated = population.clone()
    pop_size, _num_spins = updated.shape
    for indices in index_sets:
        proposed = updated.clone()
        proposed[:, indices] *= -1
        local_fields = torch.einsum("kj,ji->ki", updated, couplings[indices, :].T)
        delta_energy = -2.0 * torch.einsum("ki,ki->ki", proposed[:, indices], local_fields)
        acceptance_prob = torch.exp(-beta * delta_energy)
        random_values = torch.rand(pop_size, len(indices), device=updated.device)
        accept = (delta_energy < 0) | (random_values < acceptance_prob)
        updated[:, indices] = torch.where(accept, proposed[:, indices], updated[:, indices])
    return updated


def greedy_update(population: torch.Tensor, couplings: torch.Tensor) -> tuple[torch.Tensor, bool]:
    updated = population.clone()
    _pop_size, num_spins = updated.shape
    for index in range(num_spins):
        proposed = updated.clone()
        proposed[:, index] *= -1
        local_field = torch.einsum("kj,j->k", updated, couplings[index, :])
        delta_energy = -2.0 * proposed[:, index] * local_field
        acceptance_prob = torch.sign(-delta_energy) / 2.0 + 0.5
        if torch.sum(acceptance_prob) == 0:
            return updated, False
        accept = acceptance_prob.bool()
        updated[:, index] = torch.where(accept, proposed[:, index], updated[:, index])
    return updated, True


def checkerboard_indices_2d(length: int) -> list[torch.Tensor]:
    even = []
    odd = []
    for x in range(length):
        for y in range(length):
            index = x * length + y
            if (x + y) % 2 == 0:
                even.append(index)
            else:
                odd.append(index)
    return [torch.tensor(even), torch.tensor(odd)]


def checkerboard_indices_3d(length: int) -> list[torch.Tensor]:
    even = []
    odd = []
    for x in range(length):
        for y in range(length):
            for z in range(length):
                index = x * length * length + y * length + z
                if (x + y + z) % 2 == 0:
                    even.append(index)
                else:
                    odd.append(index)
    return [torch.tensor(even), torch.tensor(odd)]

