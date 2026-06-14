#!/usr/bin/env python
"""Generate all solver baselines for solvers_v2.

This script regenerates baseline JSON files that capture expected solver behavior.
Only run this when intentionally changing expected solver outputs.

Usage:
    PYTHONPATH=. python scripts/generate_all_baselines.py
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_baseline(family: str, algorithm: str, fixture: Path, parameters: dict, seed: int, baseline_id: str):
    """Generate a single baseline."""
    if family == "sk":
        from solvers_v2.families import sk
        runner = sk
    elif family == "ea2d":
        from solvers_v2.families.ea import ea2d
        runner = ea2d
    elif family == "ea3d":
        from solvers_v2.families.ea import ea3d
        runner = ea3d
    elif family == "rrg":
        from solvers_v2.families import rrg
        runner = rrg
    else:
        raise ValueError(f"Unknown family: {family}")

    print(f"Running {family} {algorithm} on {fixture.name}...")
    result = runner.run_baseline_case(algorithm, fixture, parameters, seed) if family != "rrg" else runner.run_pairwise_case(algorithm, fixture, parameters, seed)

    baseline = {
        "algorithm": algorithm,
        "baseline_id": baseline_id,
        "environment": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "family": family,
        "fixture": str(fixture.relative_to(ROOT)),
        "fixture_sha256": compute_sha256(fixture),
        "matrix_mode": "sk_legacy_upper" if family == "sk" else "symmetric",
        "metrics": result.metrics,
        "parameters": parameters,
        "schedule_length": result.schedule_length,
        "schema_version": 1,
        "seed_values": {
            "numpy": seed,
            "python_random": seed,
            "torch": seed,
        },
        "solver_file": f"solvers_v2/families/{family}/__init__.py",
    }

    output_file = ROOT / f"tests_data/solver_baselines/{baseline_id}.json"
    output_file.write_text(json.dumps(baseline, indent=2) + "\n")

    print(f"  ✅ {baseline_id}.json")
    print(f"     Schedule: {result.schedule_length}, Best: {baseline['metrics']['best_min_energy']:.6f}\n")


def main():
    seed = 1729

    # EA2D Population Annealing
    print("=== EA2D Population Annealing ===")
    generate_baseline(
        family="ea2d",
        algorithm="population_annealing",
        fixture=ROOT / "tests_data/instances/ea2d/N100/ea2d_couplings_N100_J0_seed2011730.txt",
        parameters={
            "L": 10,
            "pop_size": 8,
            "MCsteps": 2,
            "Tstart": 1.92,
            "Tend": 0.1,
            "num_temps": 4,
            "schedule": "linearT",
            "high_temp_thermalization_steps": 1,
        },
        seed=seed,
        baseline_id="ea2d_population_annealing",
    )

    # RRG baselines (use same parameters as SK for consistency)
    rrg_fixture = ROOT / "tests_data/instances/rrg/N50/rrg_couplings_N50_J0_seed4051730.txt"

    # Check if RRG fixture exists
    if not rrg_fixture.exists():
        print(f"⚠️  RRG fixture not found: {rrg_fixture}")
        print("   Skipping RRG baselines. Generate RRG instances first.\n")
        return

    print("=== RRG Simulated Annealing ===")
    generate_baseline(
        family="rrg",
        algorithm="simulated_annealing",
        fixture=rrg_fixture,
        parameters={
            "pop_size": 8,
            "MCsteps": 2,
            "Tstart": 1.92,
            "Tend": 0.1,
            "num_temps": 4,
            "schedule": "linearT",
            "high_temp_thermalization_steps": 1,
        },
        seed=seed,
        baseline_id="rrg_simulated_annealing",
    )

    print("=== RRG Population Annealing ===")
    generate_baseline(
        family="rrg",
        algorithm="population_annealing",
        fixture=rrg_fixture,
        parameters={
            "pop_size": 8,
            "MCsteps": 2,
            "Tstart": 1.92,
            "Tend": 0.1,
            "num_temps": 4,
            "schedule": "linearT",
            "high_temp_thermalization_steps": 1,
        },
        seed=seed,
        baseline_id="rrg_population_annealing",
    )

    print("=== RRG Parallel Tempering ===")
    generate_baseline(
        family="rrg",
        algorithm="parallel_tempering",
        fixture=rrg_fixture,
        parameters={
            "MCsteps": 2,
            "swap_interval": 1,
            "Tstart": 1.92,
            "Tend": 0.1,
            "num_temps": 4,
            "schedule": "linearT",
            "high_temp_thermalization_steps": 1,
        },
        seed=seed,
        baseline_id="rrg_parallel_tempering",
    )

    print("=== RRG Greedy ===")
    generate_baseline(
        family="rrg",
        algorithm="greedy",
        fixture=rrg_fixture,
        parameters={
            "pop_size": 8,
        },
        seed=seed,
        baseline_id="rrg_greedy",
    )

    print("✅ All missing baselines generated!")


if __name__ == "__main__":
    main()
