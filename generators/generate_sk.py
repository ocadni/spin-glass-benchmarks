import math

import numpy as np


def sample_couplings(rng, count, mean=0.0, std=1.0, distribution="gaussian"):
    if distribution == "gaussian":
        return rng.normal(mean, std, size=count)
    if distribution in {"rademacher", "radamacher"}:
        return mean + std * rng.choice([-1.0, 1.0], size=count)
    raise ValueError("distribution must be 'gaussian' or 'rademacher'")


def generate_sk(N, mean_j=0.0, distribution="gaussian", field=0.0, seed=None):
    """
    Generate a Sherrington-Kirkpatrick instance on N spins.

    Couplings are sampled with standard deviation 1/sqrt(N).
    """
    if N < 1:
        raise ValueError("N must be positive")

    rng = np.random.default_rng(seed)
    num_couplings = N * (N - 1) // 2
    values = sample_couplings(
        rng,
        num_couplings,
        mean=mean_j,
        std=1.0 / math.sqrt(N),
        distribution=distribution,
    )

    couplings = []
    value_index = 0
    for i in range(N):
        for j in range(i + 1, N):
            couplings.append((i, j, float(values[value_index])))
            value_index += 1

    fields = [(i, float(field)) for i in range(N)]
    return fields, couplings
