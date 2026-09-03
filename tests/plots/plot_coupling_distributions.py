#!/usr/bin/env python3
import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "spin_glass_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from load_couplings import load_couplings_from_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures" / "coupling_distributions_5x5.png"


def discover_coupling_files(benchmarks_dir, limit=25):
    benchmarks_dir = Path(benchmarks_dir)
    model_dirs = ["sk", "ea2d", "ea3d", "rrg"]
    grouped = []
    for model_dir in model_dirs:
        grouped.append(sorted((benchmarks_dir / model_dir).rglob("*_couplings_*.txt")))

    selected = []
    index = 0
    while len(selected) < limit:
        added = False
        for group in grouped:
            if index < len(group):
                selected.append(group[index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        index += 1

    if selected:
        return selected
    return sorted(benchmarks_dir.rglob("*_couplings_*.txt"))[:limit]


def gaussian_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2.0 * math.pi))


def title_for(coupling_set):
    metadata = coupling_set.metadata
    if metadata:
        model = metadata.get("model", "?")
        n_spins = metadata.get("N", "?")
        seed = metadata.get("seed", "?")
        title = f"{model}, N={n_spins}, seed={seed}"
    else:
        title = coupling_set.path.stem

    expected_mean, expected_std = expected_moments_for(coupling_set)
    return f"{title}\nexpected mean={expected_mean:g}, std={expected_std:.4g}"


def expected_moments_for(coupling_set):
    metadata = coupling_set.metadata
    model = metadata.get("model", "")
    mean = float(metadata.get("meanJ", 0.0))
    if model == "sk":
        std = 1.0 / math.sqrt(float(metadata["N"]))
    else:
        std = 1.0
    return mean, std


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a 5x5 plot of coupling distributions from benchmark files."
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=ROOT / "benchmarks",
        help="Benchmark root used when --file is not provided.",
    )
    parser.add_argument(
        "--file",
        action="append",
        type=Path,
        default=None,
        help="Coupling file to plot. Can be passed up to 25 times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output figure path.",
    )
    parser.add_argument("--bins", type=int, default=40, help="Histogram bin count.")
    parser.add_argument("--dpi", type=int, default=150, help="Saved figure resolution.")
    return parser.parse_args()


def main():
    args = parse_args()
    files = args.file if args.file is not None else discover_coupling_files(args.benchmarks_dir)
    files = files[:25]
    if not files:
        raise FileNotFoundError(f"No coupling files found under {args.benchmarks_dir}")

    fig, axes = plt.subplots(5, 5, figsize=(15, 12), constrained_layout=True)
    axes = axes.ravel()

    for axis, path in zip(axes, files):
        coupling_set = load_couplings_from_file(path)
        values = coupling_set.interactions[:, 2]
        axis.hist(values, bins=args.bins, density=True, color="#4c78a8", alpha=0.75)

        if coupling_set.metadata.get("distribution") == "gaussian":
            mean = float(np.mean(values))
            std = float(np.std(values))
            x_values = np.linspace(float(np.min(values)), float(np.max(values)), 300)
            axis.plot(x_values, gaussian_pdf(x_values, mean, std), color="#c44e52", linewidth=1.8)

        axis.set_title(title_for(coupling_set), fontsize=9)
        axis.tick_params(axis="both", labelsize=8)

    for axis in axes[len(files):]:
        axis.axis("off")

    fig.suptitle("Benchmark Coupling Distributions", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
