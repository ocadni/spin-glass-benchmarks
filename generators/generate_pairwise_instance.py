#!/usr/bin/env python3
"""Generate one canonical pairwise instance under instances/."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .generate_ea import save_ea
    from .generate_rrg import save_rrg
    from .generate_sk import save_sk
else:
    from generate_ea import save_ea
    from generate_rrg import save_rrg
    from generate_sk import save_sk


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
    common = {
        "mean_j": args.mean_j,
        "distribution": args.distribution,
        "field": args.field,
        "seed": args.seed,
        "outdir": args.outdir,
        "include_family_dir": True,
    }
    if args.family == "sk":
        path = save_sk(args.N, **common)
    elif args.family == "rrg":
        path = save_rrg(args.N, degree=args.degree, **common)
    else:
        dim = 2 if args.family == "ea2d" else 3
        path = save_ea(args.N, L=args.L, dim=dim, **common)
    print(path)


if __name__ == "__main__":
    main()
