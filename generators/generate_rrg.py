import argparse
import random
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


def save_rrg(
    N,
    *,
    degree=3,
    mean_j=0.0,
    distribution="gaussian",
    field=0.0,
    seed=None,
    outdir: str | Path = "instances",
    include_family_dir: bool = True,
):
    """Generate and write an RRG instance directly to disk."""
    if N < 1:
        raise ValueError("N must be positive")

    distribution = normalize_distribution(distribution)
    edges = _random_regular_edges(N, degree, seed=seed)
    rng = np.random.default_rng(seed)
    values = sample_couplings(
        rng,
        len(edges),
        mean=mean_j,
        std=1.0,
        distribution=distribution,
    )
    i_idx = np.fromiter((i for i, _ in edges), dtype=np.int64, count=len(edges))
    j_idx = np.fromiter((j for _, j in edges), dtype=np.int64, count=len(edges))
    path = instance_file_path(
        root=outdir,
        family="rrg",
        num_spins=N,
        mean_j=mean_j,
        seed=seed,
        include_family_dir=include_family_dir,
    )
    write_pairwise_text_file(
        path=path,
        family="rrg",
        num_spins=N,
        mean_j=mean_j,
        seed=seed,
        distribution=distribution,
        field=field,
        num_couplings=len(edges),
        extra_metadata={
            "graph": "random_regular",
            "degree": degree,
            "graph_seed": "none" if seed is None else seed,
        },
    )
    append_couplings(path, i_idx, j_idx, values)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a random regular graph instance.")
    parser.add_argument("N", type=int, help="Number of spins.")
    parser.add_argument("--degree", "-k", type=int, default=3, help="RRG degree.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--outdir", type=Path, default=Path("instances"))
    parser.add_argument("--meanJ", "--mean-j", dest="mean_j", type=float, default=0.0)
    parser.add_argument("--distribution", choices=["gaussian", "rademacher", "radamacher"], default="gaussian")
    parser.add_argument("--field", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    path = save_rrg(
        args.N,
        degree=args.degree,
        mean_j=args.mean_j,
        distribution=args.distribution,
        field=args.field,
        seed=args.seed,
        outdir=args.outdir,
    )
    print(path)


if __name__ == "__main__":
    main()
