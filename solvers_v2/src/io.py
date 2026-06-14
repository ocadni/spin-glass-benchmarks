from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
GENERATORS_DIR = ROOT / "generators"
if str(GENERATORS_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATORS_DIR))

from pairwise_io import load_pairwise_instance  # noqa: E402


def load_pairwise_couplings(path: str | Path, symmetric: bool) -> torch.Tensor:
    instance = load_pairwise_instance(path)
    couplings = torch.zeros(instance.num_spins, instance.num_spins)
    for interaction in instance.interactions:
        couplings[interaction.i, interaction.j] = interaction.coupling
        if symmetric:
            couplings[interaction.j, interaction.i] = interaction.coupling
    return couplings

