#!/usr/bin/env python3
"""Copy reference pairwise instances into the canonical instances/ tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from pairwise_io import benchmark_files, load_pairwise_instance, write_pairwise_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize canonical instances from reference data.")
    parser.add_argument("--source", type=Path, default=Path("tests_data/instances"))
    parser.add_argument("--target", type=Path, default=Path("instances"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def target_path(source_file: Path, source_root: Path, target_root: Path) -> Path:
    return target_root / source_file.relative_to(source_root)


def materialize(source_root: Path, target_root: Path, overwrite: bool = False) -> list[Path]:
    written: list[Path] = []
    for source_file in benchmark_files(source_root):
        instance = load_pairwise_instance(source_file)
        output_path = target_path(source_file, source_root, target_root)
        if output_path.exists() and not overwrite:
            existing = load_pairwise_instance(output_path)
            if existing.instance_hash() != instance.instance_hash():
                raise FileExistsError(
                    f"{output_path} already exists with different instance contents"
                )
            continue
        write_pairwise_instance(instance, output_path)
        written.append(output_path)
    return written


def main() -> None:
    args = parse_args()
    written = materialize(args.source, args.target, overwrite=args.overwrite)
    print(f"materialized={len(written)} target={args.target}")


if __name__ == "__main__":
    main()
