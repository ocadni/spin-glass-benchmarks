from __future__ import annotations

from pathlib import Path

import torch

from generators.pairwise_io import load_pairwise_instance


def load_pairwise_couplings(path: str | Path, symmetric: bool) -> torch.Tensor:
    instance = load_pairwise_instance(path)
    couplings = torch.zeros(instance.num_spins, instance.num_spins)
    for interaction in instance.interactions:
        couplings[interaction.i, interaction.j] = interaction.coupling
        if symmetric:
            couplings[interaction.j, interaction.i] = interaction.coupling
    return couplings
