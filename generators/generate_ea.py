import numpy as np


def sample_couplings(rng, count, mean=0.0, std=1.0, distribution="gaussian"):
    if distribution == "gaussian":
        return rng.normal(mean, std, size=count)
    if distribution in {"rademacher", "radamacher"}:
        return mean + std * rng.choice([-1.0, 1.0], size=count)
    raise ValueError("distribution must be 'gaussian' or 'rademacher'")


def _linear_index(coords, L):
    index = 0
    for coord in coords:
        index = index * L + coord
    return index


def _coordinates(index, L, dim):
    coords = [0] * dim
    for axis in range(dim - 1, -1, -1):
        coords[axis] = index % L
        index //= L
    return coords


def generate_ea(N, dim=2, mean_j=0.0, distribution="gaussian", field=0.0, seed=None):
    """
    Generate a 2D or 3D Edwards-Anderson lattice instance on N spins.

    N must be a perfect square for dim=2 or a perfect cube for dim=3.
    Periodic boundary conditions are used. Couplings have standard deviation 1.
    """
    if dim not in {2, 3}:
        raise ValueError("dim must be 2 or 3")
    if N < 1:
        raise ValueError("N must be positive")

    L = round(N ** (1.0 / dim))
    if L**dim != N:
        raise ValueError(f"N must be a perfect {dim}D hypercube size, got N={N}")

    edges = []
    seen = set()
    for i in range(N):
        coords = _coordinates(i, L, dim)
        for axis in range(dim):
            neighbor = list(coords)
            neighbor[axis] = (neighbor[axis] + 1) % L
            j = _linear_index(neighbor, L)
            edge = (min(i, j), max(i, j))
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)

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
