import argparse
import math
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

    i_idx, j_idx = np.triu_indices(N, k=1)
    couplings = list(zip(i_idx.tolist(), j_idx.tolist(), values.tolist()))

    fields = [(i, float(field)) for i in range(N)]
    return fields, couplings


def save_sk(
    N,
    *,
    mean_j=0.0,
    distribution="gaussian",
    field=0.0,
    seed=None,
    outdir: str | Path = "instances",
    include_family_dir: bool = True,
):
    """Generate and write an SK instance without materializing Python edge objects."""
    if N < 1:
        raise ValueError("N must be positive")

    distribution = normalize_distribution(distribution)
    rng = np.random.default_rng(seed)
    num_couplings = N * (N - 1) // 2
    values = sample_couplings(
        rng,
        num_couplings,
        mean=mean_j,
        std=1.0 / math.sqrt(N),
        distribution=distribution,
    )
    path = instance_file_path(
        root=outdir,
        family="sk",
        num_spins=N,
        mean_j=mean_j,
        seed=seed,
        include_family_dir=include_family_dir,
    )
    write_pairwise_text_file(
        path=path,
        family="sk",
        num_spins=N,
        mean_j=mean_j,
        seed=seed,
        distribution=distribution,
        field=field,
        num_couplings=num_couplings,
    )
    i_idx, j_idx = np.triu_indices(N, k=1)
    append_couplings(path, i_idx, j_idx, values)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Sherrington-Kirkpatrick instance.")
    parser.add_argument("N", type=int, help="Number of spins.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--outdir", type=Path, default=Path("instances"))
    parser.add_argument("--meanJ", "--mean-j", dest="mean_j", type=float, default=0.0)
    parser.add_argument("--distribution", choices=["gaussian", "rademacher", "radamacher"], default="gaussian")
    parser.add_argument("--field", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    path = save_sk(
        args.N,
        mean_j=args.mean_j,
        distribution=args.distribution,
        field=args.field,
        seed=args.seed,
        outdir=args.outdir,
    )
    print(path)


if __name__ == "__main__":
    main()
