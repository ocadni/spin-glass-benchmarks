import random

import numpy as np


def _try_generate_sets(K, rng):
    rows = [list(range(K)) for _ in range(3)]
    for row in rows:
        rng.shuffle(row)

    triples = []
    seen_triples = set()
    used_pairs = set()

    for a, b, c in zip(*rows):
        triple = tuple(sorted((a, b, c)))
        if len(set(triple)) < 3:
            return None
        if triple in seen_triples:
            return None

        p, q, k = triple
        pairs = ((p, q), (p, k), (q, k))
        if any(pair in used_pairs for pair in pairs):
            return None

        triples.append(triple)
        seen_triples.add(triple)
        used_pairs.update(pairs)

    return triples


def generate_sets(K, seed=None, max_attempts=100000):
    """
    Return K sorted 3-tuples with no repeated variables and no repeated pairs.

    K must satisfy K = 1 mod 6 or K = 3 mod 6.
    """
    if not isinstance(K, int) or K < 7:
        raise ValueError("K must be an integer >= 7")
    if K % 6 not in {1, 3}:
        raise ValueError(f"K must satisfy K = 1 mod 6 or K = 3 mod 6, got {K}")

    rng = random.Random(seed)
    for _ in range(max_attempts):
        triples = _try_generate_sets(K, rng)
        if triples is not None:
            return triples

    raise RuntimeError(f"Could not generate valid XORSAT triples for K={K}")


def generate_xorsat(K, field=0.0, seed=None):
    """
    Generate a satisfiable 3-XORSAT Ising instance.

    K is the number of physical variables and clauses. The returned Ising
    instance has 2K spins because each clause gets one auxiliary spin.
    """
    triples = generate_sets(K, seed=seed)

    b = []
    pair_couplings = {}
    fields = {i: float(field) for i in range(2 * K)}

    for clause_index, (p, q, k) in enumerate(triples):
        aux = K + clause_index
        clause_b = 0
        b.append(clause_b)

        sign = 1
        physical_coupling = -1.0
        auxiliary_coupling = -2.0
        physical_field = sign * 1.0
        auxiliary_field = sign * 2.0

        for edge, value in (
            ((p, q), physical_coupling),
            ((p, k), physical_coupling),
            ((q, k), physical_coupling),
            ((p, aux), auxiliary_coupling),
            ((q, aux), auxiliary_coupling),
            ((k, aux), auxiliary_coupling),
        ):
            edge = tuple(sorted(edge))
            pair_couplings[edge] = pair_couplings.get(edge, 0.0) + value

        fields[p] += physical_field
        fields[q] += physical_field
        fields[k] += physical_field
        fields[aux] += auxiliary_field

    field_list = [(i, fields[i]) for i in range(2 * K)]
    coupling_list = [
        (i, j, value) for (i, j), value in sorted(pair_couplings.items())
    ]
    metadata = {
        "K": K,
        "triples": triples,
        "b": b,
    }
    return field_list, coupling_list, metadata


def generate_XORSAT(K, rng=None):
    """
    Compatibility wrapper for older code.

    Returns dictionaries for couplings and fields, plus the XORSAT triples,
    right-hand sides, and an all-ones satisfying physical spin solution.
    """
    seed = rng if isinstance(rng, int) else None
    fields, couplings, metadata = generate_xorsat(K, seed=seed)
    pairs = {(i, j): value for i, j, value in couplings}
    h = {i: value for i, value in fields if value != 0.0}
    return (
        pairs,
        h,
        metadata["triples"],
        np.array(metadata["b"]),
        np.ones(K),
    )
