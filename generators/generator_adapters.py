"""Adapters from existing generator functions to canonical pairwise instances."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from pairwise_instance import PairwiseInstance


ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ROOT / "generators"
if str(GENERATORS) not in sys.path:
    sys.path.insert(0, str(GENERATORS))

from generate_ea import generate_ea  # noqa: E402
from generate_rrg import generate_rrg  # noqa: E402
from generate_sk import generate_sk  # noqa: E402


def generate_pairwise_instance(
    family: str,
    params: Mapping[str, Any],
    seed: int | None = None,
) -> PairwiseInstance:
    family = family.lower()
    if family == "sk":
        return generate_sk_instance(params, seed=seed)
    if family in {"ea2d", "ea3d"}:
        dim = 2 if family == "ea2d" else 3
        return generate_ea_instance(params, dim=dim, seed=seed)
    if family == "rrg":
        return generate_rrg_instance(params, seed=seed)
    raise ValueError(f"unsupported pairwise family: {family}")


def generate_sk_instance(params: Mapping[str, Any], seed: int | None = None) -> PairwiseInstance:
    num_spins = _required_int(params, "N")
    mean_j = float(params.get("mean_j", params.get("meanJ", 0.0)))
    distribution = _distribution(params)
    field = float(params.get("field", 0.0))
    fields, couplings = generate_sk(
        num_spins,
        mean_j=mean_j,
        distribution=distribution,
        field=field,
        seed=seed,
    )
    return _instance("sk", num_spins, fields, couplings, mean_j, distribution, field, seed)


def generate_ea_instance(
    params: Mapping[str, Any],
    dim: int,
    seed: int | None = None,
) -> PairwiseInstance:
    num_spins = int(params["L"]) ** dim if "L" in params else _required_int(params, "N")
    mean_j = float(params.get("mean_j", params.get("meanJ", 0.0)))
    distribution = _distribution(params)
    field = float(params.get("field", 0.0))
    fields, couplings = generate_ea(
        num_spins,
        dim=dim,
        mean_j=mean_j,
        distribution=distribution,
        field=field,
        seed=seed,
    )
    metadata = {"dim": dim}
    return _instance(
        f"ea{dim}d",
        num_spins,
        fields,
        couplings,
        mean_j,
        distribution,
        field,
        seed,
        extra_metadata=metadata,
    )


def generate_rrg_instance(params: Mapping[str, Any], seed: int | None = None) -> PairwiseInstance:
    num_spins = _required_int(params, "N")
    degree = int(params.get("degree", 3))
    mean_j = float(params.get("mean_j", params.get("meanJ", 0.0)))
    distribution = _distribution(params)
    field = float(params.get("field", 0.0))
    fields, couplings = generate_rrg(
        num_spins,
        degree=degree,
        mean_j=mean_j,
        distribution=distribution,
        field=field,
        seed=seed,
    )
    return _instance(
        "rrg",
        num_spins,
        fields,
        couplings,
        mean_j,
        distribution,
        field,
        seed,
        extra_metadata={
            "graph": "random_regular",
            "degree": degree,
            "graph_seed": "none" if seed is None else seed,
        },
    )


def _instance(
    family: str,
    num_spins: int,
    fields: list[tuple[int, float]],
    couplings: list[tuple[int, int, float]],
    mean_j: float,
    distribution: str,
    field: float,
    seed: int | None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> PairwiseInstance:
    metadata: dict[str, Any] = {
        "model": family,
        "N": num_spins,
        "meanJ": _filename_number(mean_j),
        "seed": "none" if seed is None else seed,
        "distribution": distribution,
        "field": _filename_number(field),
        "num_fields": len(fields),
        "num_couplings": len(couplings),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return PairwiseInstance.from_edges(
        family=family,
        num_spins=num_spins,
        fields=fields,
        couplings=couplings,
        metadata=metadata,
    )


def _distribution(params: Mapping[str, Any]) -> str:
    distribution = str(params.get("distribution", "gaussian"))
    return "rademacher" if distribution == "radamacher" else distribution


def _required_int(params: Mapping[str, Any], key: str) -> int:
    if key not in params:
        raise ValueError(f"missing required parameter: {key}")
    return int(params[key])


def _filename_number(value: float) -> str:
    text = f"{float(value):.5f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text
