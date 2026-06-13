from __future__ import annotations

from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_io import benchmark_files, load_pairwise_instance, write_pairwise_instance


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests_data" / "instances"
TARGET = ROOT / "instances"


def deterministic_spins(num_spins: int, offset: int = 0) -> list[int]:
    return [1 if (index + offset) % 2 == 0 else -1 for index in range(num_spins)]


def test_instances_are_semantically_equal_to_reference_data(tmp_path):
    reference_files = benchmark_files(REFERENCE)
    if not reference_files:
        pytest.skip("no benchmark reference files found")
    if not TARGET.exists():
        pytest.skip("instances/ has not been materialized yet")

    missing = []
    for reference_file in reference_files:
        generated_file = TARGET / reference_file.relative_to(REFERENCE)
        if not generated_file.exists():
            missing.append(generated_file)
            continue

        source_instance = load_pairwise_instance(reference_file)
        generated_instance = load_pairwise_instance(generated_file)

        assert generated_instance.family == source_instance.family
        assert generated_instance.num_spins == source_instance.num_spins
        assert generated_instance.fields == source_instance.fields
        assert generated_instance.interactions == source_instance.interactions
        assert generated_instance.instance_hash() == source_instance.instance_hash()

        for offset in range(3):
            spins = deterministic_spins(source_instance.num_spins, offset=offset)
            assert generated_instance.energy(spins) == pytest.approx(source_instance.energy(spins))

        round_trip = tmp_path / generated_file.relative_to(TARGET)
        write_pairwise_instance(generated_instance, round_trip)
        assert load_pairwise_instance(round_trip).instance_hash() == generated_instance.instance_hash()

    assert missing == []
