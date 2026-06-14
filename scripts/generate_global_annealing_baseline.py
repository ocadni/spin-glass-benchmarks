#!/usr/bin/env python
"""Generate baseline for SK global annealing."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

# Import after setting up paths
from solvers_v2.families import sk


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    fixture = ROOT / "tests_data/instances/sk/N50/sk_couplings_N50_J0_seed1051730.txt"

    # Use small parameters for fast baseline generation
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

    print(f"Running global annealing on {fixture.name}...")
    result = sk.run_baseline_case("global_annealing", fixture, parameters, seed)

    baseline = {
        "algorithm": "global_annealing",
        "baseline_id": "sk_global_annealing",
        "environment": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "family": "sk",
        "fixture": str(fixture.relative_to(ROOT)),
        "fixture_sha256": compute_sha256(fixture),
        "matrix_mode": "sk_legacy_upper",
        "metrics": result.metrics,
        "parameters": parameters,
        "schedule_length": result.schedule_length,
        "schema_version": 1,
        "seed_values": {
            "numpy": seed,
            "python_random": seed,
            "torch": seed,
        },
        "solver_file": "solvers_v2/common/global_annealing/__init__.py",
    }

    output_file = ROOT / "tests_data/solver_baselines/sk_global_annealing.json"
    output_file.write_text(json.dumps(baseline, indent=2) + "\n")

    print(f"✅ Baseline written to {output_file.relative_to(ROOT)}")
    print(f"   Schedule length: {result.schedule_length}")
    print(f"   Best min energy: {baseline['metrics']['best_min_energy']:.6f}")
    print(f"   Final min energy: {baseline['metrics']['final_min_energy']:.6f}")


if __name__ == "__main__":
    main()
