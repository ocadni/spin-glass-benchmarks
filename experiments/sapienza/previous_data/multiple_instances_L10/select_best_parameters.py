#!/usr/bin/env python3
"""Reduce GA_alternative/PA_alternative summaries to each seed's best TTS row.

Reads GA_alternative_summary.csv and PA_alternative_summary.csv (produced by
summarize_alternative_results.py), which hold one row per (seed, parameter
set) with num_runs, success_rate, average_time and a tts_p<target> column.
For each seed, keeps only the row with the lowest TTS (ties broken by higher
success_rate, then lower average_time, then the parameter values themselves,
for a deterministic result). Seeds where every parameter set has TTS == inf
are still included, using the same tie-break rule, so every seed present in
the input appears exactly once in the output.

Usage:
    python3 select_best_parameters.py
Writes GA_alternative_best_summary.csv and PA_alternative_best_summary.csv
next to this script.
"""

from __future__ import annotations

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
        return (
            tts,
            -float(row["success_rate"]),
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


def main() -> int:
    exit_code = 0

    for algorithm in ALGORITHMS:
        summary_path = SCRIPT_DIR / f"{algorithm}_summary.csv"

        if not summary_path.exists():
            print(f"error: summary file not found: {summary_path}", file=sys.stderr)
            exit_code = 1
            continue

        best_rows, fieldnames = select_best_rows(summary_path)

        output_path = SCRIPT_DIR / f"{algorithm}_best_summary.csv"
        write_best_summary(output_path, best_rows, fieldnames)

        unsolved = sum(1 for row in best_rows if not math.isfinite(float(row[find_tts_column(fieldnames)])))
        print(
            f"wrote {len(best_rows)} seeds to {output_path} "
            f"({unsolved} with no successful run at any parameter set)"
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
