#!/usr/bin/env python3
"""Append GA_alternative/PA_alternative best-per-seed results to the website summary.csv.

Reads GA_alternative_best_summary.csv and PA_alternative_best_summary.csv
(produced by select_best_parameters.py: one row per seed, that algorithm's
best-TTS parameter set) and appends them, reformatted to the website's
summary.csv schema (see experiments/sapienza/code/greedy_solvers/analizer.py),
to experiments/sapienza/results/ea3d/summary.csv:

    N, seed, min_energy, average_time, success_probability, TTS, hardware,
    program_name, average_steps, parameters

N is fixed at 1000 (these previous_data instances are EA3D, linear size
L=10). average_steps is left blank (not tracked by these solvers).
parameters is a free-text rendering of the winning parameter set, shown only
in the website's "All" section (the Leaderboard ignores it).

This script only ever appends; re-running it after the target already
contains these rows will duplicate them. Usage:
    python3 publish_to_website_summary.py
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
TARGET_SUMMARY_CSV = REPO_ROOT / "experiments" / "sapienza" / "results" / "ea3d" / "summary.csv"

N = 1000
HARDWARE = "NVIDIA Tesla V100-SXM2-32G"

PROGRAM_NAMES = {
    "GA_alternative": "Global Annealing",
    "PA_alternative": "Population Annealing",
}

NON_PARAM_COLUMNS = {"seed", "num_runs", "best_energy", "success_rate", "average_time"}

TARGET_FIELDNAMES = [
    "N", "seed", "min_energy", "average_time", "success_probability",
    "TTS", "hardware", "program_name", "average_steps", "parameters",
]


def find_tts_column(fieldnames: list[str]) -> str:
    tts_columns = [name for name in fieldnames if name.startswith("tts")]

    if len(tts_columns) != 1:
        raise ValueError(f"expected exactly one tts_* column, found {tts_columns}")

    return tts_columns[0]


def format_parameters(row: dict[str, str], param_columns: list[str]) -> str:
    return ", ".join(f"{name}={row[name]}" for name in param_columns)


def load_website_rows(best_summary_path: Path, program_name: str) -> list[dict[str, str]]:
    with best_summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    tts_column = find_tts_column(fieldnames)
    param_columns = [name for name in fieldnames if name not in NON_PARAM_COLUMNS and name != tts_column]

    website_rows = []
    for row in rows:
        tts = float(row[tts_column])
        website_rows.append(
            {
                "N": N,
                "seed": row["seed"],
                "min_energy": row["best_energy"],
                "average_time": row["average_time"],
                "success_probability": row["success_rate"],
                "TTS": f"{tts:.12g}" if math.isfinite(tts) else "inf",
                "hardware": HARDWARE,
                "program_name": program_name,
                "average_steps": "",
                "parameters": format_parameters(row, param_columns),
            }
        )

    return website_rows


def upgrade_header_if_needed(path: Path, required_fields: list[str]) -> list[str]:
    """Add any of required_fields missing from the file's header, preserving
    the existing column order and backfilling old rows with blanks. Returns
    the resulting (possibly unchanged) header."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing_fields = reader.fieldnames or []
        missing_fields = [name for name in required_fields if name not in existing_fields]
        if not missing_fields:
            return existing_fields
        rows = list(reader)

    fieldnames = [*existing_fields, *missing_fields]

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.chmod(path.stat().st_mode & 0o777)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return fieldnames


def append_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writerows(rows)


def main() -> int:
    if not TARGET_SUMMARY_CSV.exists():
        print(f"error: target summary.csv not found: {TARGET_SUMMARY_CSV}", file=sys.stderr)
        return 1

    fieldnames = upgrade_header_if_needed(TARGET_SUMMARY_CSV, TARGET_FIELDNAMES)

    all_rows: list[dict[str, str]] = []
    for algorithm, program_name in PROGRAM_NAMES.items():
        best_summary_path = SCRIPT_DIR / f"{algorithm}_best_summary.csv"

        if not best_summary_path.exists():
            print(f"error: best summary not found: {best_summary_path}", file=sys.stderr)
            return 1

        rows = load_website_rows(best_summary_path, program_name)
        all_rows.extend(rows)
        print(f"prepared {len(rows)} rows for {program_name}")

    append_rows(TARGET_SUMMARY_CSV, all_rows, fieldnames)
    print(f"appended {len(all_rows)} rows to {TARGET_SUMMARY_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
