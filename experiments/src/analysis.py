"""Utilities for loading and analyzing experiment results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from experiments.src.config import ExperimentConfig
from experiments.src.results import CommonResult, DiagnosticsResult


@dataclass
class ExperimentResults:
    """Container for experiment results with analysis utilities."""

    config: ExperimentConfig
    experiment_dir: Path

    def get_algorithm_results(self, algorithm: str) -> list[CommonResult]:
        """Load all results for a specific algorithm."""
        algo_dir = self.experiment_dir / algorithm / "runs"
        if not algo_dir.exists():
            return []

        results = []

        for run_dir in algo_dir.glob("*_seed*"):
            common_path = run_dir / "common.json"
            if common_path.exists():
                results.append(CommonResult.from_file(common_path))

        return results

    def get_diagnostics(self, algorithm: str) -> list[DiagnosticsResult]:
        """Load all diagnostics for a specific algorithm."""
        algo_dir = self.experiment_dir / algorithm / "runs"
        if not algo_dir.exists():
            return []

        diagnostics = []

        for run_dir in algo_dir.glob("*_seed*"):
            diag_path = run_dir / "diagnostics.json"
            if diag_path.exists():
                diagnostics.append(DiagnosticsResult.from_file(diag_path))

        return diagnostics

    def iter_runs(
        self, algorithm: str
    ) -> Iterator[tuple[CommonResult, DiagnosticsResult | None]]:
        """Iterate over all runs, yielding (common, diagnostics) pairs."""
        algo_dir = self.experiment_dir / algorithm / "runs"
        if not algo_dir.exists():
            return

        for run_dir in algo_dir.glob("*_seed*"):
            common_path = run_dir / "common.json"
            diag_path = run_dir / "diagnostics.json"

            if common_path.exists():
                common = CommonResult.from_file(common_path)
                diag = (
                    DiagnosticsResult.from_file(diag_path) if diag_path.exists() else None
                )
                yield common, diag


def load_experiment(experiment_path: Path) -> ExperimentResults:
    """Load experiment from directory path."""
    meta_path = experiment_path / "experiment_meta.json"
    config = ExperimentConfig.from_file(meta_path)
    return ExperimentResults(config, experiment_path)
