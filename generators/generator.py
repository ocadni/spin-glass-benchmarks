import argparse
from pathlib import Path

from generate_ea import generate_ea
from generate_rrg import generate_rrg
from generate_sk import generate_sk


def _format_number(value):
    return f"{value:g}"


def _write_instance(path, model, N, mean_j, seed, distribution, field, fields, couplings):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# "
            f"model={model} N={N} meanJ={mean_j:g} seed={seed} "
            f"distribution={distribution} field={field:g} "
            f"num_fields={len(fields)} num_couplings={len(couplings)}\n"
        )
        for index, value in fields:
            handle.write(f"{index} {value:.17g}\n")
        for i, j, value in couplings:
            handle.write(f"{i} {j} {value:.17g}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate spin glass benchmark instances."
    )
    parser.add_argument(
        "model",
        choices=["sk", "ea", "rrg"],
        help="Model to generate.",
    )
    parser.add_argument(
        "N",
        type=int,
        nargs="?",
        help="Number of spins. Required for SK/RRG; optional for EA if --L is given.",
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

    output_model = args.model

    if args.model == "sk":
        fields, couplings = generate_sk(
            N,
            mean_j=args.mean_j,
            distribution=distribution,
            field=args.field,
            seed=args.seed,
        )
    elif args.model == "ea":
        output_model = f"ea{args.dim}d"
        fields, couplings = generate_ea(
            N,
            dim=args.dim,
            mean_j=args.mean_j,
            distribution=distribution,
            field=args.field,
            seed=args.seed,
        )
    else:
        fields, couplings = generate_rrg(
            N,
            degree=args.degree,
            mean_j=args.mean_j,
            distribution=distribution,
            field=args.field,
            seed=args.seed,
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{output_model}_couplings_N{N}_"
        f"J{_format_number(args.mean_j)}_seed{args.seed}.txt"
    )
    path = outdir / filename
    _write_instance(
        path,
        output_model,
        N,
        args.mean_j,
        args.seed,
        distribution,
        args.field,
        fields,
        couplings,
    )
    print(path)


if __name__ == "__main__":
    main()
