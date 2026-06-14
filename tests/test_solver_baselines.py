from __future__ import annotations

import pytest


pytest.importorskip("torch")

from solver_baseline_harness import baseline_specs, compare_with_baseline  # noqa: E402


@pytest.mark.parametrize("spec", baseline_specs(), ids=lambda spec: spec.baseline_id)
def test_old_solver_baselines_match_recorded_metrics(spec):
    compare_with_baseline(spec)
