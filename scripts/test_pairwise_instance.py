from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_instance import Interaction, PairwiseInstance


def test_energy_with_fields_uses_documented_sign_convention():
    instance = PairwiseInstance.from_edges(
        family="unit",
        num_spins=3,
        fields=[(0, 0.5), (1, -1.0), (2, 0.0)],
        couplings=[(0, 1, 2.0), (1, 2, -3.0)],
    )

    assert instance.energy([1, -1, 1]) == pytest.approx(4.5)


def test_interactions_are_sorted_and_reversed_rows_are_canonicalized():
    instance = PairwiseInstance.from_edges(
        family="unit",
        num_spins=4,
        couplings=[(3, 1, 0.25), (0, 2, -1.0)],
    )

    assert instance.interactions == (
        Interaction(0, 2, -1.0),
        Interaction(1, 3, 0.25),
    )


def test_dense_couplings_can_be_symmetric_or_upper_triangular():
    instance = PairwiseInstance.from_edges(
        family="unit",
        num_spins=3,
        couplings=[(0, 2, 1.5)],
    )

    assert instance.dense_couplings(symmetric=False) == [
        [0.0, 0.0, 1.5],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert instance.dense_couplings(symmetric=True)[2][0] == 1.5


def test_instance_hash_is_stable_against_input_order_and_metadata():
    first = PairwiseInstance.from_edges(
        family="unit",
        num_spins=3,
        couplings=[(1, 2, 0.5), (0, 1, -2.0)],
        metadata={"seed": "1"},
    )
    second = PairwiseInstance.from_edges(
        family="unit",
        num_spins=3,
        couplings=[(0, 1, -2.0), (2, 1, 0.5)],
        metadata={"seed": "different"},
    )

    assert first.instance_hash() == second.instance_hash()


def test_invalid_spin_assignment_is_rejected():
    instance = PairwiseInstance.from_edges("unit", 2, [(0, 1, 1.0)])

    with pytest.raises(ValueError, match="spins must be"):
        instance.energy([1, 0])
