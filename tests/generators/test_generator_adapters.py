from __future__ import annotations

from generators.generate_ea import generate_ea
from generators.generate_rrg import generate_rrg
from generators.generate_sk import generate_sk
from generators.generator_adapters import generate_pairwise_instance


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
        assert [(x.i, x.j, x.coupling) for x in instance.interactions] == sorted(couplings)


def test_rrg_adapter_matches_existing_generator():
    fields, couplings = generate_rrg(10, degree=3, field=-0.5, seed=13)

    instance = generate_pairwise_instance("rrg", {"N": 10, "degree": 3, "field": -0.5}, seed=13)

    assert instance.fields == tuple(value for _, value in fields)
    assert [(x.i, x.j, x.coupling) for x in instance.interactions] == couplings
    assert instance.metadata["graph"] == "random_regular"
    assert instance.metadata["degree"] == 3
    assert instance.metadata["graph_seed"] == 13
