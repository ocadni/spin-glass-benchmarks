import argparse
from pathlib import Path

if __package__:
    from .generate_ea import save_ea
    from .generate_rrg import save_rrg
    from .generate_sk import save_sk
    from .generate_xorsat import save_xorsat
else:
    from generate_ea import save_ea
    from generate_rrg import save_rrg
    from generate_sk import save_sk
    from generate_xorsat import save_xorsat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate spin glass benchmark instances."
    )
    parser.add_argument(
        "model",
        choices=["sk", "ea", "rrg", "xorsat"],
        help="Model to generate.",
    )
    parser.add_argument(
        "N",
        type=int,
        nargs="?",
        help=(
            "Number of spins for SK/RRG, physical variables K for XORSAT, "
            "or optional EA spins if --L is given."
        ),
    )
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory where the instance file is written.",
    )
    parser.add_argument(
        "--meanJ",
        "--mean-j",
        dest="mean_j",
        type=float,
        default=0.0,
        help="Mean coupling value.",
    )
    parser.add_argument(
        "--distribution",
        choices=["gaussian", "rademacher", "radamacher"],
        default="gaussian",
        help="Coupling distribution.",
    )
    parser.add_argument(
        "--field",
        type=float,
        default=0.0,
        help="Constant external field on every spin.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=[2, 3],
        default=None,
        help="EA lattice dimension.",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=None,
        help="EA lattice linear size. If provided, N is set to L**dim.",
    )
    parser.add_argument(
        "--degree",
        "-k",
        type=int,
        default=None,
        help="RRG degree.",
    )
    args = parser.parse_args()

    if args.model != "ea" and args.L is not None:
        parser.error("--L can only be used with the EA model")
    if args.model != "ea" and args.dim is not None:
        parser.error("--dim can only be used with the EA model")
    if args.model != "rrg" and args.degree is not None:
        parser.error("--degree/-k can only be used with the RRG model")
    if args.model == "xorsat" and args.distribution != "gaussian":
        parser.error("--distribution is not used by the XORSAT model")
    if args.model == "xorsat" and args.mean_j != 0.0:
        parser.error("--meanJ/--mean-j is not used by the XORSAT model")

    if args.model == "ea":
        args.dim = 2 if args.dim is None else args.dim
    elif args.model == "rrg":
        args.degree = 3 if args.degree is None else args.degree

    return args


def main():
    args = parse_args()
    distribution = "rademacher" if args.distribution == "radamacher" else args.distribution
    N = args.N

    if args.model == "ea" and args.L is not None:
        if args.L < 1:
            raise ValueError("L must be positive")
        if args.N is not None:
            raise ValueError("For EA, specify either positional N or --L, not both")
        N = args.L**args.dim

    if N is None:
        raise ValueError(f"N is required for model '{args.model}'")

    outdir = Path(args.outdir)
    common = {
        "field": args.field,
        "seed": args.seed,
        "outdir": outdir,
        "include_family_dir": False,
    }
    if args.model == "sk":
        path = save_sk(
            N,
            mean_j=args.mean_j,
            distribution=distribution,
            **common,
        )
    elif args.model == "ea":
        path = save_ea(
            N,
            dim=args.dim,
            mean_j=args.mean_j,
            distribution=distribution,
            **common,
        )
    elif args.model == "rrg":
        path = save_rrg(
            N,
            degree=args.degree,
            mean_j=args.mean_j,
            distribution=distribution,
            **common,
        )
    else:
        path = save_xorsat(
            N,
            **common,
        )
    print(path)


if __name__ == "__main__":
    main()
