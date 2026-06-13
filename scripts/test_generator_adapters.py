from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ROOT / "generators"
if str(GENERATORS) not in sys.path:
    sys.path.insert(0, str(GENERATORS))

from generate_ea import generate_ea  # noqa: E402
from generate_rrg import generate_rrg  # noqa: E402
from generate_sk import generate_sk  # noqa: E402
from generator_adapters import generate_pairwise_instance  # noqa: E402


def test_sk_adapter_matches_existing_generator():
    fields, couplings = generate_sk(8, mean_j=0.2, distribution="rademacher", field=0.1, seed=5)

    instance = generate_pairwise_instance(
        "sk",
        {"N": 8, "mean_j": 0.2, "distribution": "rademacher", "field": 0.1},
        seed=5,
    )

    assert instance.fields == tuple(value for _, value in fields)
    assert [(x.i, x.j, x.coupling) for x in instance.interactions] == couplings


def test_ea_adapters_match_existing_generator():
    for family, dim in [("ea2d", 2), ("ea3d", 3)]:
        fields, couplings = generate_ea(3**dim, dim=dim, seed=11)
        instance = generate_pairwise_instance(family, {"N": 3**dim}, seed=11)

        assert instance.fields == tuple(value for _, value in fields)
        assert [(x.i, x.j, x.coupling) for x in instance.interactions] == couplings


def test_rrg_adapter_matches_existing_generator():
    fields, couplings = generate_rrg(10, degree=3, field=-0.5, seed=13)

    instance = generate_pairwise_instance("rrg", {"N": 10, "degree": 3, "field": -0.5}, seed=13)

    assert instance.fields == tuple(value for _, value in fields)
    assert [(x.i, x.j, x.coupling) for x in instance.interactions] == couplings
    assert instance.metadata["graph"] == "random_regular"
    assert instance.metadata["degree"] == 3
    assert instance.metadata["graph_seed"] == 13
