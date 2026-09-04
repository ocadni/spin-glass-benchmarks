from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def schedule_temperatures(
    t_start: float,
    t_end: float,
    num_temps_determiner: int | float | Sequence[float],
    schedule: str,
    num_spins: int,
) -> np.ndarray | list[float]:
    if schedule == "linearT":
        return np.linspace(t_start, t_end, int(num_temps_determiner))
    if schedule == "linearBeta":
        return [1.0 / beta for beta in np.linspace(1.0 / t_start, 1.0 / t_end, int(num_temps_determiner))]
    if schedule == "logT":
        return np.logspace(np.log10(t_start), np.log10(t_end), num=int(num_temps_determiner))
    if schedule == "custom":
        return list(num_temps_determiner)  # type: ignore[arg-type]
    if schedule == "Cv_beta":
        return [1.0 / beta for beta in get_betas_3d(1.0 / t_start, 1.0 / t_end, float(num_temps_determiner), num_spins)]
    raise ValueError(f"invalid temperature schedule: {schedule}")


def get_betas_3d(start: float, finish: float, factor: float, num_spins: int) -> np.ndarray:
    beta = start
    betas = [beta]
    while beta < finish:
        beta += factor / np.sqrt(_cv_3d(1.0 / beta, num_spins))
        betas.append(beta)
    return np.array(betas)


def _cv_3d(temperature: float, num_spins: int) -> float:
    return (
        num_spins
        * 34.19
        * temperature
        * temperature
        / (10.36 + temperature * temperature * temperature)
        / (10.36 + temperature * temperature * temperature)
    )

