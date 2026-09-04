import argparse
from pathlib import Path

import numpy as np

if __package__:
    from ._file_writer import (
        append_couplings,
        instance_file_path,
        normalize_distribution,
        write_pairwise_text_file,
    )
else:
    from _file_writer import (
        append_couplings,
        instance_file_path,
        normalize_distribution,
        write_pairwise_text_file,
    )


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


def save_ea(
    N=None,
    *,
    L=None,
    dim=2,
    mean_j=0.0,
    distribution="gaussian",
    field=0.0,
    seed=None,
    outdir: str | Path = "instances",
    include_family_dir: bool = True,
):
    """Generate and write an EA lattice instance directly to disk."""
    if L is not None:
        if L < 1:
            raise ValueError("L must be positive")
        if N is not None:
            raise ValueError("For EA, specify either positional N or --L, not both")
        N = L**dim
    if N is None:
        raise ValueError("N is required")

    distribution = normalize_distribution(distribution)
    fields, couplings = generate_ea(
        N,
        dim=dim,
        mean_j=mean_j,
        distribution=distribution,
        field=field,
        seed=seed,
    )
    couplings = sorted(couplings)
    i_idx = np.fromiter((i for i, _, _ in couplings), dtype=np.int64, count=len(couplings))
    j_idx = np.fromiter((j for _, j, _ in couplings), dtype=np.int64, count=len(couplings))
    values = np.fromiter((value for _, _, value in couplings), dtype=np.float64, count=len(couplings))

    family = f"ea{dim}d"
    path = instance_file_path(
        root=outdir,
        family=family,
        num_spins=N,
        mean_j=mean_j,
        seed=seed,
        include_family_dir=include_family_dir,
    )
    write_pairwise_text_file(
        path=path,
        family=family,
        num_spins=N,
        mean_j=mean_j,
        seed=seed,
        distribution=distribution,
        field=field,
        num_couplings=len(couplings),
        extra_metadata={"dim": dim},
    )
    append_couplings(path, i_idx, j_idx, values)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an Edwards-Anderson lattice instance.")
    parser.add_argument("N", type=int, nargs="?", help="Number of spins.")
    parser.add_argument("--L", type=int, help="Lattice side length. If provided, N is L**dim.")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2)
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--outdir", type=Path, default=Path("instances"))
    parser.add_argument("--meanJ", "--mean-j", dest="mean_j", type=float, default=0.0)
    parser.add_argument("--distribution", choices=["gaussian", "rademacher", "radamacher"], default="gaussian")
    parser.add_argument("--field", type=float, default=0.0)
    args = parser.parse_args()
    if args.N is None and args.L is None:
        parser.error("N or --L is required")
    return args


def main():
    args = parse_args()
    path = save_ea(
        args.N,
        L=args.L,
        dim=args.dim,
        mean_j=args.mean_j,
        distribution=args.distribution,
        field=args.field,
        seed=args.seed,
        outdir=args.outdir,
    )
    print(path)


if __name__ == "__main__":
    main()
