#!/usr/bin/env python3
"""Reduce GA_alternative/PA_alternative summaries to each seed's best TTS row.

Reads GA_alternative_summary.csv and PA_alternative_summary.csv (produced by
summarize_alternative_results.py), which hold one row per (seed, parameter
set) with num_runs, best_energy, success_rate, average_time and a
tts_p<target> column. For each seed, keeps only the row with the lowest TTS
(ties broken by higher success_rate, then the parameter values themselves,
for a deterministic result). Seeds where every parameter set has TTS == inf
(success_rate == 0, i.e. no parameter set ever solved that seed) are still
included: among those, the tie-break is instead the lowest best_energy found
(then average_time, then the parameter values), so the reported "best"
failed run is the one that got closest to the ground state rather than
merely the fastest. Every seed present in the input appears exactly once in
the output.

Usage:
    python3 select_best_parameters.py [--data-dir DIR]
Writes GA_alternative_best_summary.csv and PA_alternative_best_summary.csv
into --data-dir (default: this script's own directory).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ALGORITHMS = ("GA_alternative", "PA_alternative")

NON_PARAM_COLUMNS = {"seed", "num_runs", "best_energy", "success_rate", "average_time"}


def find_tts_column(fieldnames: list[str]) -> str:
    tts_columns = [name for name in fieldnames if name.startswith("tts")]

    if len(tts_columns) != 1:
        raise ValueError(f"expected exactly one tts_* column, found {tts_columns}")

    return tts_columns[0]


def coerce_param(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def select_best_rows(summary_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    tts_column = find_tts_column(fieldnames)
    param_columns = [
        name for name in fieldnames
        if name not in NON_PARAM_COLUMNS and not name.startswith("tts")
    ]

    best_by_seed: dict[int, dict[str, str]] = {}

    def sort_key(row: dict[str, str]) -> tuple:
        tts = float(row[tts_column])
        # When nothing solved this (seed, parameter set) — tts is inf — TTS
        # and success_rate can't distinguish parameter sets, so fall back to
        # the lowest energy found rather than the fastest run.
        third_field = float(row["best_energy"]) if not math.isfinite(tts) else float(row["average_time"])
        return (
            tts,
            -float(row["success_rate"]),
            third_field,
            float(row["average_time"]),
            tuple(coerce_param(row[name]) for name in param_columns),
        )

    for row in rows:
        seed = int(row["seed"])
        current_best = best_by_seed.get(seed)

        if current_best is None or sort_key(row) < sort_key(current_best):
            best_by_seed[seed] = row

    best_rows = [best_by_seed[seed] for seed in sorted(best_by_seed)]

    return best_rows, fieldnames


def write_best_summary(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce GA_alternative/PA_alternative summaries to each seed's best TTS row."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR,
        help=f"directory containing the *_summary.csv files (default: {SCRIPT_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exit_code = 0

    for algorithm in ALGORITHMS:
        summary_path = args.data_dir / f"{algorithm}_summary.csv"

        if not summary_path.exists():
            print(f"error: summary file not found: {summary_path}", file=sys.stderr)
            exit_code = 1
            continue

        best_rows, fieldnames = select_best_rows(summary_path)

        output_path = args.data_dir / f"{algorithm}_best_summary.csv"
        write_best_summary(output_path, best_rows, fieldnames)

        unsolved = sum(1 for row in best_rows if not math.isfinite(float(row[find_tts_column(fieldnames)])))
        print(
            f"wrote {len(best_rows)} seeds to {output_path} "
            f"({unsolved} with no successful run at any parameter set)"
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
