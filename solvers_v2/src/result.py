from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from solvers_v2.src.observables import Observables


@dataclass(frozen=True)
class SolverResult:
    schedule: Any
    observables: Observables

    @property
    def metrics(self) -> dict[str, float | list[float]]:
        min_energy = self.observables.get_observable_history("min_energy")
        mean_energy = self.observables.get_observable_history("mean_energy")
        return {
            "min_energy": min_energy,
            "mean_energy": mean_energy,
            "final_min_energy": min_energy[-1],
            "final_mean_energy": mean_energy[-1],
            "best_min_energy": min(min_energy),
        }

    @property
    def schedule_length(self) -> int | None:
        if self.schedule is None:
            return None
        return len(self.schedule)

