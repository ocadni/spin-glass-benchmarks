"""2D Edwards-Anderson solver adapters."""
from __future__ import annotations

from pathlib import Path

from solvers_v2.src.result import SolverResult
from solvers_v2.families.ea.common import run_baseline_case as _run_ea_baseline_case


def run_baseline_case(algorithm: str, fixture: str | Path, parameters: dict, seed: int, progress_callback=None) -> SolverResult:
    return _run_ea_baseline_case(algorithm, fixture, parameters, seed, dimension="2d", progress_callback=progress_callback)

