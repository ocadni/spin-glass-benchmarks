#!/usr/bin/env python3
"""Generate one canonical pairwise instance under instances/."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .generator_adapters import generate_pairwise_instance
    from .pairwise_io import default_instance_path, write_pairwise_instance
else:
    from generator_adapters import generate_pairwise_instance
    from pairwise_io import default_instance_path, write_pairwise_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a canonical V0 pairwise instance.")
    parser.add_argument("family", choices=["sk", "ea2d", "ea3d", "rrg"])
    parser.add_argument("N", type=int, nargs="?", help="Number of spins.")
    parser.add_argument("--L", type=int, help="EA lattice side length.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("instances"))
    parser.add_argument("--meanJ", "--mean-j", dest="mean_j", type=float, default=0.0)
    parser.add_argument("--distribution", choices=["gaussian", "rademacher", "radamacher"], default="gaussian")
    parser.add_argument("--field", type=float, default=0.0)
    parser.add_argument("--degree", type=int, default=3, help="RRG degree.")
    args = parser.parse_args()

    if args.L is not None and args.family not in {"ea2d", "ea3d"}:
        parser.error("--L can only be used with ea2d or ea3d")
    if args.family in {"sk", "rrg"} and args.N is None:
        parser.error("N is required for sk and rrg")
    if args.family in {"ea2d", "ea3d"} and args.N is None and args.L is None:
        parser.error("N or --L is required for ea2d and ea3d")
    if args.family != "rrg" and args.degree != parser.get_default("degree"):
        parser.error("--degree can only be used with rrg")
    return args


def main() -> None:
    args = parse_args()
    params = {
        "mean_j": args.mean_j,
        "distribution": args.distribution,
        "field": args.field,
    }
    if args.L is not None:
        params["L"] = args.L
    else:
        params["N"] = args.N
    if args.family == "rrg":
        params["degree"] = args.degree

    instance = generate_pairwise_instance(args.family, params, seed=args.seed)
    path = default_instance_path(instance, args.outdir)
    write_pairwise_instance(instance, path)
    print(path)


if __name__ == "__main__":
    main()
