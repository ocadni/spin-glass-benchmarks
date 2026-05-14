import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ROOT / "generators"
sys.path.insert(0, str(GENERATORS))

from generate_ea import generate_ea  # noqa: E402
from generate_rrg import generate_rrg  # noqa: E402
from generate_sk import generate_sk  # noqa: E402


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
        "0.2",
    )

    assert result.returncode == 0, result.stderr
    output_path = tmp_path / "ea2d_couplings_N9_J0_seed17.txt"
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# model=ea2d N=9 meanJ=0 seed=17")
    assert lines[1] == "0 0.20000000000000001"


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
