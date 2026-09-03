from __future__ import annotations

from pathlib import Path

import pytest

from generators.pairwise_io import benchmark_files


ROOT = Path(__file__).resolve().parents[2]
GENERATED_ROOT = ROOT / "instances"
REFERENCE_ROOT = ROOT / "tests/data" / "instances"
REFERENCE_FILES_BY_RELATIVE_PATH = {
    path.relative_to(REFERENCE_ROOT): path for path in benchmark_files(REFERENCE_ROOT)
}
GENERATED_FILES_BY_RELATIVE_PATH = {
    path.relative_to(GENERATED_ROOT): path for path in benchmark_files(GENERATED_ROOT)
}


def test_reference_instance_fixture_set_is_not_empty():
    assert REFERENCE_FILES_BY_RELATIVE_PATH


def test_generated_instance_file_set_matches_reference_snapshot():
    reference_paths = set(REFERENCE_FILES_BY_RELATIVE_PATH)
    generated_paths = set(GENERATED_FILES_BY_RELATIVE_PATH)

    missing = sorted(reference_paths - generated_paths)
    unexpected = sorted(generated_paths - reference_paths)

    assert not missing and not unexpected, (
        "instances/ does not match tests/data/instances/.\n"
        f"Missing files: {_format_paths(missing)}\n"
        f"Unexpected files: {_format_paths(unexpected)}"
    )


@pytest.mark.parametrize(
    "relative_path",
    sorted(REFERENCE_FILES_BY_RELATIVE_PATH),
    ids=str,
)
def test_generated_instance_file_matches_reference_snapshot(relative_path):
    reference_path = REFERENCE_FILES_BY_RELATIVE_PATH[relative_path]
    generated_path = GENERATED_ROOT / relative_path

    assert generated_path.exists()
    assert generated_path.read_bytes() == reference_path.read_bytes()


def _format_paths(paths: list[Path], limit: int = 10) -> str:
    if not paths:
        return "none"
    shown = ", ".join(str(path) for path in paths[:limit])
    remaining = len(paths) - limit
    if remaining > 0:
        shown = f"{shown}, ... ({remaining} more)"
    return shown
