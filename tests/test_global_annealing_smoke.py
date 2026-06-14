"""Smoke test for global annealing to verify it runs without errors."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def test_sk_global_annealing_smoke():
    """Verify global annealing runs on a small SK instance."""
    from solvers_v2.families import sk

    fixture = ROOT / "tests_data/instances/sk/N50/sk_couplings_N50_J0_seed1051730.txt"
    parameters = {
        "pop_size": 100,
        "MLMCsteps": 2,
        "swap_step": 2,
        "Tstart": 1.92,
        "Tend": 0.1,
        "num_temps": 4,
        "schedule": "linearT",
        "high_temp_thermalization_steps": 1,
        "num_epochs_start": 5,
        "num_epochs_retrain": 1,
        "batch_size": 50,
    }
    seed = 1729

    result = sk.run_baseline_case("global_annealing", fixture, parameters, seed)

    assert result is not None
    assert result.schedule_length == 4
    assert "min_energy" in result.metrics
    assert "mean_energy" in result.metrics
    assert len(result.metrics["min_energy"]) == 4
    assert len(result.metrics["mean_energy"]) == 4
