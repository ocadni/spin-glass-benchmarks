"""Core pairwise Ising instance representation for V0 benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


PROBLEM_TYPE = "pairwise_ising"
ENERGY_CONVENTION = "H(s)=-sum_{i<j}J_ij*s_i*s_j-sum_i h_i*s_i"
SPIN_VALUES = {-1, 1}


def canonical_float(value: float) -> str:
    """Return a stable string representation for hashing and canonical JSON."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite float value: {value!r}")
    if number == 0.0:
        return "0"
    return format(number, ".17g")


def _validate_interactions(
    interactions_in: tuple["Interaction", ...], num_spins: int
) -> tuple["Interaction", ...]:
    """Validate edges and return them sorted with no duplicates.

    Generators emit edges already sorted with i < j, so this checks that in a
    single linear pass and only pays for a full sort + dedup when the input
    isn't already in canonical order (e.g. hand-built or loaded out of order).
    """
    prev_edge: tuple[int, int] | None = None
    already_sorted = True
    for interaction in interactions_in:
        if not isinstance(interaction, Interaction):
            raise TypeError("interactions must contain Interaction objects")
        if interaction.j >= num_spins:
            raise ValueError("interaction index out of range")
        edge = (interaction.i, interaction.j)
        if prev_edge is not None and edge <= prev_edge:
            already_sorted = False
            break
        prev_edge = edge

    if already_sorted:
        return tuple(interactions_in)

    interactions = tuple(sorted(interactions_in))
    seen_edges: set[tuple[int, int]] = set()
    for interaction in interactions:
        if not isinstance(interaction, Interaction):
            raise TypeError("interactions must contain Interaction objects")
        if interaction.j >= num_spins:
            raise ValueError("interaction index out of range")
        edge = (interaction.i, interaction.j)
        if edge in seen_edges:
            raise ValueError(f"duplicate interaction edge: {edge}")
        seen_edges.add(edge)
    return interactions


def _freeze_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if values is None:
        return MappingProxyType({})
    return MappingProxyType(dict(values))


def _canonical_metadata(values: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): str(values[key]) for key in sorted(values)}


@dataclass(frozen=True, order=True)
class Interaction:
    """A single pairwise coupling J_ij with zero-based indices and i < j."""

    i: int
    j: int
    coupling: float

    def __post_init__(self) -> None:
        if not isinstance(self.i, int) or not isinstance(self.j, int):
            raise TypeError("interaction indices must be integers")
        if self.i < 0 or self.j < 0:
            raise ValueError("interaction indices must be non-negative")
        if self.i >= self.j:
            raise ValueError("interactions must use canonical order i < j")
        if not math.isfinite(float(self.coupling)):
            raise ValueError("coupling must be finite")
        object.__setattr__(self, "coupling", float(self.coupling))

    def canonical_data(self) -> list[Any]:
        return [self.i, self.j, canonical_float(self.coupling)]


@dataclass(frozen=True)
class PairwiseInstance:
    """Sparse pairwise Ising instance with optional longitudinal fields."""

    family: str
    num_spins: int
    interactions: tuple[Interaction, ...]
    fields: tuple[float, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    normalization: Mapping[str, Any] = field(default_factory=dict)
    problem_type: str = PROBLEM_TYPE
    energy_convention: str = ENERGY_CONVENTION

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("family must be non-empty")
        if self.num_spins < 1:
            raise ValueError("num_spins must be positive")
        if self.problem_type != PROBLEM_TYPE:
            raise ValueError(f"unsupported problem_type: {self.problem_type!r}")

        fields = tuple(float(value) for value in self.fields) if self.fields else (0.0,) * self.num_spins
        if len(fields) != self.num_spins:
            raise ValueError("fields length must equal num_spins")
        if any(not math.isfinite(value) for value in fields):
            raise ValueError("fields must be finite")

        interactions = _validate_interactions(self.interactions, self.num_spins)

        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "interactions", interactions)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "normalization", _freeze_mapping(self.normalization))

    @classmethod
    def from_edges(
        cls,
        family: str,
        num_spins: int,
        couplings: Iterable[tuple[int, int, float]],
        fields: Sequence[float] | Mapping[int, float] | Iterable[tuple[int, float]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        normalization: Mapping[str, Any] | None = None,
    ) -> "PairwiseInstance":
        return cls(
            family=family,
            num_spins=num_spins,
            interactions=tuple(_interaction_from_row(row) for row in couplings),
            fields=fields_to_vector(num_spins, fields),
            metadata=metadata or {},
            normalization=normalization or {},
        )

    def validate_spins(self, spins: Sequence[int]) -> tuple[int, ...]:
        if len(spins) != self.num_spins:
            raise ValueError("spin assignment length must equal num_spins")
        assignment = tuple(int(spin) for spin in spins)
        invalid = [spin for spin in assignment if spin not in SPIN_VALUES]
        if invalid:
            raise ValueError("spins must be in {-1, +1}")
        return assignment

    def energy(self, spins: Sequence[int]) -> float:
        assignment = self.validate_spins(spins)
        pair_energy = sum(
            interaction.coupling * assignment[interaction.i] * assignment[interaction.j]
            for interaction in self.interactions
        )
        field_energy = sum(field * spin for field, spin in zip(self.fields, assignment))
        return float(-pair_energy - field_energy)

    def dense_couplings(self, symmetric: bool = True) -> list[list[float]]:
        matrix = [[0.0 for _ in range(self.num_spins)] for _ in range(self.num_spins)]
        for interaction in self.interactions:
            matrix[interaction.i][interaction.j] = interaction.coupling
            if symmetric:
                matrix[interaction.j][interaction.i] = interaction.coupling
        return matrix

    def canonical_data(self, include_metadata: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "problem_type": self.problem_type,
            "family": self.family,
            "num_spins": self.num_spins,
            "energy_convention": self.energy_convention,
            "normalization": _canonical_metadata(self.normalization),
            "fields": [canonical_float(value) for value in self.fields],
            "interactions": [interaction.canonical_data() for interaction in self.interactions],
        }
        if include_metadata:
            data["metadata"] = _canonical_metadata(self.metadata)
        return data

    def instance_hash(self) -> str:
        payload = json.dumps(
            self.canonical_data(include_metadata=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()


def fields_to_vector(
    num_spins: int,
    fields: Sequence[float] | Mapping[int, float] | Iterable[tuple[int, float]] | None,
) -> tuple[float, ...]:
    if fields is None:
        return (0.0,) * num_spins

    if isinstance(fields, Mapping):
        vector = [0.0] * num_spins
        items = fields.items()
    else:
        sequence = list(fields)
        if len(sequence) == num_spins and all(not _looks_like_field_row(value) for value in sequence):
            return tuple(float(value) for value in sequence)  # type: ignore[arg-type]
        vector = [0.0] * num_spins
        items = sequence  # type: ignore[assignment]

    for index, value in items:
        spin_index = int(index)
        if spin_index < 0 or spin_index >= num_spins:
            raise ValueError(f"field index out of range: {spin_index}")
        vector[spin_index] = float(value)
    return tuple(vector)


def _looks_like_field_row(value: Any) -> bool:
    return isinstance(value, (tuple, list)) and len(value) == 2


def _interaction_from_row(row: tuple[int, int, float]) -> Interaction:
    i, j, coupling = row
    left = int(i)
    right = int(j)
    if left == right:
        raise ValueError("self-couplings are not supported")
    if left > right:
        left, right = right, left
    return Interaction(left, right, float(coupling))
