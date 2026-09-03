import subprocess
import sys
from collections import Counter
from math import sqrt
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]

from generators.generate_ea import generate_ea
from generators.generate_rrg import generate_rrg
from generators.generate_sk import generate_sk
from generators.generate_xorsat import generate_sets, generate_xorsat


GENERATORS = ROOT / "generators"


def coupling_values(couplings):
    return np.array([value for _, _, value in couplings])


def assert_coupling_distribution(
    values,
    expected_mean,
    expected_std,
    expected_third_cumulant=0.0,
    expected_fourth_cumulant=0.0,
    mean_tolerance=0.05,
    std_tolerance=0.05,
    third_cumulant_tolerance=0.2,
    fourth_cumulant_tolerance=0.4,
):
    mean = float(np.mean(values))
    std = float(np.std(values))
    centered = values - mean
    third_cumulant = float(np.mean(centered**3))
    fourth_cumulant = float(np.mean(centered**4) - 3.0 * np.var(values) ** 2)

    assert mean == pytest.approx(expected_mean, abs=mean_tolerance * expected_std)
    assert std == pytest.approx(expected_std, rel=std_tolerance)
    assert third_cumulant == pytest.approx(
        expected_third_cumulant,
        abs=third_cumulant_tolerance * expected_std**3,
    )
    assert fourth_cumulant == pytest.approx(
        expected_fourth_cumulant,
        abs=fourth_cumulant_tolerance * expected_std**4,
    )


def test_generate_sk_complete_graph_and_fields():
    fields, couplings = generate_sk(5, field=0.25, seed=123)

    assert fields == [(i, 0.25) for i in range(5)]
    assert len(couplings) == 10
    assert {(i, j) for i, j, _ in couplings} == {
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    }


def test_generate_sk_is_reproducible_for_same_seed():
    assert generate_sk(8, seed=7) == generate_sk(8, seed=7)


def test_generate_sk_rademacher_values_include_sk_scaling_and_mean():
    _, couplings = generate_sk(4, mean_j=2.0, distribution="rademacher", seed=1)

    values = {value for _, _, value in couplings}
    assert values <= {1.5, 2.5}


@pytest.mark.parametrize(
    ("N", "mean_j", "distribution", "expected_fourth_cumulant"),
    [
        (350, 0.3, "gaussian", 0.0),
        (350, -0.2, "rademacher", -2.0 * (1.0 / sqrt(350)) ** 4),
    ],
)
def test_generate_sk_coupling_distribution_moments_and_cumulants(
    N,
    mean_j,
    distribution,
    expected_fourth_cumulant,
):
    _, couplings = generate_sk(N, mean_j=mean_j, distribution=distribution, seed=101)

    assert_coupling_distribution(
        coupling_values(couplings),
        expected_mean=mean_j,
        expected_std=1.0 / sqrt(N),
        expected_fourth_cumulant=expected_fourth_cumulant,
    )


@pytest.mark.parametrize(
    ("dim", "N", "expected_edges"),
    [
        (2, 9, 18),
        (3, 27, 81),
    ],
)
def test_generate_ea_periodic_lattice_edge_counts(dim, N, expected_edges):
    fields, couplings = generate_ea(N, dim=dim, field=-0.5, seed=3)

    assert fields == [(i, -0.5) for i in range(N)]
    assert len(couplings) == expected_edges
    assert len({(i, j) for i, j, _ in couplings}) == expected_edges
    assert all(0 <= i < j < N for i, j, _ in couplings)


def test_generate_ea_rejects_non_hypercube_size():
    with pytest.raises(ValueError, match="perfect 3D hypercube"):
        generate_ea(100, dim=3)


def test_generate_ea_rademacher_values_have_unit_scale():
    _, couplings = generate_ea(9, dim=2, mean_j=0.5, distribution="rademacher", seed=5)

    values = {value for _, _, value in couplings}
    assert values <= {-0.5, 1.5}


@pytest.mark.parametrize(
    ("mean_j", "distribution", "expected_fourth_cumulant"),
    [
        (-0.2, "gaussian", 0.0),
        (0.5, "rademacher", -2.0),
    ],
)
def test_generate_ea_coupling_distribution_moments_and_cumulants(
    mean_j,
    distribution,
    expected_fourth_cumulant,
):
    _, couplings = generate_ea(
        10000,
        dim=2,
        mean_j=mean_j,
        distribution=distribution,
        seed=202,
    )

    assert_coupling_distribution(
        coupling_values(couplings),
        expected_mean=mean_j,
        expected_std=1.0,
        expected_fourth_cumulant=expected_fourth_cumulant,
    )


def test_generate_rrg_degree_and_edge_count():
    N = 10
    degree = 3
    fields, couplings = generate_rrg(N, degree=degree, field=1.0, seed=11)

    assert fields == [(i, 1.0) for i in range(N)]
    assert len(couplings) == N * degree // 2
    degrees = Counter()
    for i, j, _ in couplings:
        degrees[i] += 1
        degrees[j] += 1
    assert degrees == Counter({i: degree for i in range(N)})


@pytest.mark.parametrize(
    ("N", "degree", "message"),
    [
        (5, 3, "N \\* degree must be even"),
        (4, 4, "degree must be smaller than N"),
    ],
)
def test_generate_rrg_rejects_invalid_degree(N, degree, message):
    with pytest.raises(ValueError, match=message):
        generate_rrg(N, degree=degree)


@pytest.mark.parametrize(
    ("mean_j", "distribution", "expected_fourth_cumulant"),
    [
        (0.1, "gaussian", 0.0),
        (-0.4, "rademacher", -2.0),
    ],
)
def test_generate_rrg_coupling_distribution_moments_and_cumulants(
    mean_j,
    distribution,
    expected_fourth_cumulant,
):
    _, couplings = generate_rrg(
        2000,
        degree=4,
        mean_j=mean_j,
        distribution=distribution,
        seed=303,
    )

    assert_coupling_distribution(
        coupling_values(couplings),
        expected_mean=mean_j,
        expected_std=1.0,
        expected_fourth_cumulant=expected_fourth_cumulant,
        fourth_cumulant_tolerance=0.5,
    )


def test_generate_xorsat_structure_and_fields():
    K = 7
    field = 0.25
    fields, couplings, metadata = generate_xorsat(K, field=field, seed=13)

    assert len(fields) == 2 * K
    assert len(couplings) == 6 * K
    assert metadata["K"] == K
    assert len(metadata["triples"]) == K
    assert len(metadata["b"]) == K
    assert set(metadata) == {"K", "triples", "b"}
    assert metadata["b"] == [0 for _ in range(K)]
    assert all(0 <= i < 2 * K for i, _ in fields)
    assert all(0 <= i < j < 2 * K for i, j, _ in couplings)

    auxiliary_fields = dict(fields)
    for aux in range(K, 2 * K):
        assert auxiliary_fields[aux] == field + 2.0


def test_generate_xorsat_triples_do_not_reuse_pairs():
    triples = generate_sets(9, seed=21)
    used_pairs = set()

    for triple in triples:
        assert len(triple) == 3
        assert tuple(sorted(triple)) == triple
        p, q, k = triple
        for pair in ((p, q), (p, k), (q, k)):
            assert pair not in used_pairs
            used_pairs.add(pair)


def test_generate_xorsat_is_reproducible_for_same_seed():
    assert generate_xorsat(7, seed=31) == generate_xorsat(7, seed=31)


def test_generate_xorsat_rejects_invalid_k():
    with pytest.raises(ValueError, match="K must satisfy"):
        generate_xorsat(8, seed=1)


def run_generator(*args):
    return subprocess.run(
        [sys.executable, str(GENERATORS / "generator.py"), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_writes_ea_dimension_in_filename_and_header(tmp_path):
    result = run_generator(
        "ea",
        "--L",
        "3",
        "--dim",
        "2",
        "--seed",
        "17",
        "--outdir",
        str(tmp_path),
        "--field",
        "0.234567",
    )

    assert result.returncode == 0, result.stderr
    output_path = tmp_path / "N9" / "ea2d_couplings_N9_J0_seed17.txt"
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# model=ea2d N=9 meanJ=0 seed=17")
    assert " field=0.23457 " in lines[0]
    assert lines[1] == "0 0.23457"


def test_cli_writes_xorsat_total_spin_count_and_k_metadata(tmp_path):
    result = run_generator(
        "xorsat",
        "7",
        "--seed",
        "19",
        "--outdir",
        str(tmp_path),
        "--field",
        "0.1",
    )

    assert result.returncode == 0, result.stderr
    output_path = tmp_path / "N14" / "xorsat_couplings_N14_J0_seed19.txt"
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# model=xorsat N=14 meanJ=0 seed=19")
    assert " K=7" in lines[0]
    assert len(lines) == 1 + 14 + 42


@pytest.mark.parametrize(
    "args",
    [
        ("sk", "10", "--L", "3", "--seed", "1"),
        ("rrg", "10", "--dim", "2", "--seed", "1"),
        ("ea", "--L", "3", "--degree", "4", "--seed", "1"),
    ],
)
def test_cli_rejects_model_specific_options_on_wrong_models(args):
    result = run_generator(*args)

    assert result.returncode == 2
    assert "can only be used with" in result.stderr
