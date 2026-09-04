"""3D Edwards-Anderson solver adapters."""
from __future__ import annotations

from pathlib import Path

from experiments.sapienza.src.result import SolverResult
from experiments.sapienza.families.ea.common import run_baseline_case as _run_ea_baseline_case


def run_baseline_case(algorithm: str, fixture: str | Path, parameters: dict, seed: int, progress_callback=None) -> SolverResult:
    return _run_ea_baseline_case(algorithm, fixture, parameters, seed, dimension="3d", progress_callback=progress_callback)

