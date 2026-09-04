from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "tests/data" / "solver_baselines"


def baseline_files() -> list[Path]:
    return sorted(BASELINE_DIR.glob("*.json"))


@pytest.mark.parametrize("baseline_file", baseline_files(), ids=lambda path: path.stem)
def test_solvers_v2_matches_old_solver_baselines(baseline_file):
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    runner = _runner_for_family(baseline["family"])

    # RRG uses run_pairwise_case, others use run_baseline_case
    if baseline["family"] == "rrg":
        result = runner.run_pairwise_case(
            baseline["algorithm"],
            ROOT / baseline["fixture"],
            baseline["parameters"],
            baseline["seed_values"]["torch"],
        )
    else:
        result = runner.run_baseline_case(
            baseline["algorithm"],
            ROOT / baseline["fixture"],
            baseline["parameters"],
            baseline["seed_values"]["torch"],
        )

    assert result.schedule_length == baseline["schedule_length"]
    for metric_name in ("min_energy", "mean_energy"):
        np.testing.assert_allclose(
            result.metrics[metric_name],
            baseline["metrics"][metric_name],
            rtol=0.0,
            atol=1e-7,
            err_msg=f"{baseline['baseline_id']} {metric_name}",
        )
    for metric_name in ("final_min_energy", "final_mean_energy", "best_min_energy"):
        np.testing.assert_allclose(
            result.metrics[metric_name],
            baseline["metrics"][metric_name],
            rtol=0.0,
            atol=1e-7,
        )


def _runner_for_family(family: str):
    if family == "sk":
        from experiments.sapienza.families import sk

        return sk
    if family == "ea2d":
        from experiments.sapienza.families.ea import ea2d

        return ea2d
    if family == "ea3d":
        from experiments.sapienza.families.ea import ea3d

        return ea3d
    if family == "rrg":
        from experiments.sapienza.families import rrg

        return rrg
    raise ValueError(f"no Sapienza baseline runner for family {family!r}")
