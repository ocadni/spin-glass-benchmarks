#!/usr/bin/env python3
import argparse
import re
from collections import namedtuple
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


CouplingSet = namedtuple("CouplingSet", ["path", "metadata", "fields", "interactions"])


def _format_number(value):
    return f"{value:g}"


def _parse_header(line):
    if not line.startswith("#"):
        return {}

    metadata = {}
    for item in line[1:].strip().split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        metadata[key] = value
    return metadata


def _infer_n_from_filename(path):
    match = re.search(r"_N(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Cannot infer N from filename: {path}")
    return int(match.group(1))


def find_coupling_file(benchmarks_dir, model, N, mean_j, seed):
    benchmarks_dir = Path(benchmarks_dir)
    filename = (
        f"{model}_couplings_N{N}_"
        f"J{_format_number(mean_j)}_seed{seed}.txt"
    )
    matches = sorted(benchmarks_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"No benchmark file matching {filename} under {benchmarks_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple benchmark files match {filename}: {matches}")
    return matches[0]


def load_couplings_from_file(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        raise ValueError(f"Empty coupling file: {path}")

    metadata = _parse_header(lines[0])
    data_lines = lines[1:] if metadata else lines
    n_fields = int(metadata.get("num_fields", metadata.get("N", _infer_n_from_filename(path))))

    if len(data_lines) < n_fields:
        raise ValueError(f"File has fewer data lines than expected fields: {path}")

    fields = np.array(
        [[int(parts[0]), float(parts[1])] for parts in (line.split() for line in data_lines[:n_fields])],
        dtype=float,
    )
    interactions = np.array(
        [
            [int(parts[0]), int(parts[1]), float(parts[2])]
            for parts in (line.split() for line in data_lines[n_fields:])
        ],
        dtype=float,
    )

    return CouplingSet(
        path=path,
        metadata=metadata,
        fields=fields,
        interactions=interactions,
    )


def load_couplings(benchmarks_dir, model, N, mean_j, seed):
    path = find_coupling_file(benchmarks_dir, model, N, mean_j, seed)
    return load_couplings_from_file(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load one benchmark coupling file and print interaction statistics."
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to a coupling file. If provided, model/N/meanJ/seed are ignored.",
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=ROOT / "benchmarks",
        help="Benchmark root used when selecting by model/N/meanJ/seed.",
    )
    parser.add_argument("--model", help="Model name in the filename, for example sk, ea2d, ea3d, or rrg.")
    parser.add_argument("--N", type=int, help="Number of spins in the filename.")
    parser.add_argument("--meanJ", type=float, default=0.0, help="Mean coupling in the filename.")
    parser.add_argument("--seed", type=int, help="Seed in the filename.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.file is not None:
        coupling_set = load_couplings_from_file(args.file)
    else:
        missing = [name for name in ("model", "N", "seed") if getattr(args, name) is None]
        if missing:
            raise ValueError(f"Missing required selector arguments: {', '.join(missing)}")
        coupling_set = load_couplings(
            args.benchmarks_dir,
            args.model,
            args.N,
            args.meanJ,
            args.seed,
        )

    values = coupling_set.interactions[:, 2]
    print(f"path={coupling_set.path}")
    print(f"num_fields={len(coupling_set.fields)}")
    print(f"num_interactions={len(values)}")
    print(f"mean={np.mean(values):.12g}")
    print(f"std={np.std(values):.12g}")
    if coupling_set.metadata:
        print("metadata=" + " ".join(f"{key}={value}" for key, value in coupling_set.metadata.items()))


if __name__ == "__main__":
    main()
