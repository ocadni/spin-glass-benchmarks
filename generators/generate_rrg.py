import random

import numpy as np


def sample_couplings(rng, count, mean=0.0, std=1.0, distribution="gaussian"):
    if distribution == "gaussian":
        return rng.normal(mean, std, size=count)
    if distribution in {"rademacher", "radamacher"}:
        return mean + std * rng.choice([-1.0, 1.0], size=count)
    raise ValueError("distribution must be 'gaussian' or 'rademacher'")


def _random_regular_edges(N, degree, seed=None, max_attempts=1000):
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if degree >= N:
        raise ValueError("degree must be smaller than N")
    if (N * degree) % 2 != 0:
        raise ValueError("N * degree must be even")

    if degree == 0:
        return []

    rng = random.Random(seed)
    stubs = [node for node in range(N) for _ in range(degree)]

    for _ in range(max_attempts):
        rng.shuffle(stubs)
        edges = set()
        failed = False
        for pos in range(0, len(stubs), 2):
            i = stubs[pos]
            j = stubs[pos + 1]
            if i == j:
                failed = True
                break
            edge = (min(i, j), max(i, j))
            if edge in edges:
                failed = True
                break
            edges.add(edge)
        if not failed:
            return sorted(edges)

    raise RuntimeError("could not sample a simple random regular graph")


def generate_rrg(
    N,
    degree=3,
    mean_j=0.0,
    distribution="gaussian",
    field=0.0,
    seed=None,
):
    """
    Generate a random regular graph instance on N spins.

    Couplings have standard deviation 1.
    """
    if N < 1:
        raise ValueError("N must be positive")

    edges = _random_regular_edges(N, degree, seed=seed)
    rng = np.random.default_rng(seed)
    values = sample_couplings(
        rng,
        len(edges),
        mean=mean_j,
        std=1.0,
        distribution=distribution,
    )

    fields = [(i, float(field)) for i in range(N)]
    couplings = [(i, j, float(values[index])) for index, (i, j) in enumerate(edges)]
    return fields, couplings
