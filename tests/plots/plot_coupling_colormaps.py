#!/usr/bin/env python3
import argparse
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
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures" / "coupling_colormaps_5x5.png"


def discover_coupling_files(benchmarks_dir, limit=25):
    benchmarks_dir = Path(benchmarks_dir)
    model_dirs = ["sk", "ea2d", "ea3d", "rrg", "xorsat"]
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


def coupling_matrix(coupling_set):
    n_spins = len(coupling_set.fields)
    matrix = np.zeros((n_spins, n_spins), dtype=float)
    for i, j, value in coupling_set.interactions:
        i = int(i)
        j = int(j)
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix


def title_for(coupling_set):
    metadata = coupling_set.metadata
    if metadata:
        model = metadata.get("model", "?")
        n_spins = metadata.get("N", "?")
        seed = metadata.get("seed", "?")
        return f"{model}, N={n_spins}, seed={seed}"
    return coupling_set.path.stem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a 5x5 plot of coupling matrix colormaps from benchmark files."
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
    parser.add_argument("--dpi", type=int, default=150, help="Saved figure resolution.")
    return parser.parse_args()


def main():
    args = parse_args()
    files = args.file if args.file is not None else discover_coupling_files(args.benchmarks_dir)
    files = files[:25]
    if not files:
        raise FileNotFoundError(f"No coupling files found under {args.benchmarks_dir}")

    coupling_sets = [load_couplings_from_file(path) for path in files]
    matrices = [coupling_matrix(coupling_set) for coupling_set in coupling_sets]
    max_abs = max(float(np.max(np.abs(matrix))) for matrix in matrices)
    if max_abs == 0.0:
        max_abs = 1.0

    fig, axes = plt.subplots(5, 5, figsize=(15, 12), constrained_layout=True)
    axes = axes.ravel()

    image = None
    for axis, coupling_set, matrix in zip(axes, coupling_sets, matrices):
        image = axis.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title_for(coupling_set), fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    for axis in axes[len(files):]:
        axis.axis("off")

    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), shrink=0.75, label="J")

    fig.suptitle("Benchmark Coupling Colormaps", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
