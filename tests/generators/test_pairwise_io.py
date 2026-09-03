from __future__ import annotations

from pathlib import Path

from generators.pairwise_instance import PairwiseInstance
from generators.pairwise_io import (
    format_instance_number,
    load_pairwise_instance,
    write_pairwise_instance,
)


ROOT = Path(__file__).resolve().parents[2]


def test_load_existing_benchmark_format():
    path = ROOT / "tests/data" / "instances" / "sk" / "N50" / "sk_couplings_N50_J0_seed1051730.txt"

    instance = load_pairwise_instance(path)

    assert instance.family == "sk"
    assert instance.num_spins == 50
    assert len(instance.fields) == 50
    assert len(instance.interactions) == 1225
    assert instance.metadata["seed"] == "1051730"


def test_round_trip_preserves_instance_hash_and_rows(tmp_path):
    instance = PairwiseInstance.from_edges(
        family="sk",
        num_spins=3,
        fields=[(0, 0.25), (1, 0.0), (2, -0.25)],
        couplings=[(0, 1, 0.125), (1, 2, -1.5)],
        metadata={
            "model": "sk",
            "N": "3",
            "meanJ": "0",
            "seed": "7",
            "distribution": "gaussian",
            "field": "0",
        },
    )
    path = tmp_path / "sk_couplings_N3_J0_seed7.txt"

    write_pairwise_instance(instance, path)
    loaded = load_pairwise_instance(path)

    assert loaded.instance_hash() == instance.instance_hash()
    assert loaded.fields == instance.fields
    assert loaded.interactions == instance.interactions


def test_writer_creates_parent_directories(tmp_path):
    instance = PairwiseInstance.from_edges("rrg", 2, [(0, 1, 1.0)])
    path = tmp_path / "rrg" / "N2" / "rrg_couplings_N2_J0_seed1.txt"

    write_pairwise_instance(instance, path)

    assert path.exists()


def test_writer_uses_legacy_five_decimal_format(tmp_path):
    assert format_instance_number(0.234567) == "0.23457"
    assert format_instance_number(-0.000001) == "0"
    assert format_instance_number(1.2) == "1.2"

    instance = PairwiseInstance.from_edges(
        family="sk",
        num_spins=2,
        fields=[(0, 0.234567), (1, 0.0)],
        couplings=[(0, 1, -0.123456)],
    )
    path = tmp_path / "sk_couplings_N2_J0_seed1.txt"

    write_pairwise_instance(instance, path)

    assert path.read_text(encoding="utf-8").splitlines()[1:] == [
        "0 0.23457",
        "1 0",
        "0 1 -0.12346",
    ]


def test_rrg_loader_infers_graph_generation_metadata():
    path = ROOT / "tests/data" / "instances" / "rrg" / "N50" / "rrg_couplings_N50_J0_seed4051730.txt"

    instance = load_pairwise_instance(path)

    assert instance.metadata["graph"] == "random_regular"
    assert instance.metadata["degree"] == "3"
    assert instance.metadata["graph_seed"] == "4051730"
