#!/usr/bin/env python3
"""Summarize GA_alternative/PA_alternative results into per-parameter-set statistics.

Each results_<seed>.txt file under GA_alternative/ or PA_alternative/ holds one row
per run: a leading block of run-parameter columns (4 for GA_alternative, 3 for
PA_alternative) followed by three fixed columns:
    <params...> best_energy_found final_energy elapsed_time

best_energies.txt (seed, best_energy) gives the reference ground-truth minimum
energy for each seed. This script groups runs by (seed, parameter set) and, for
each group, reports the number of runs, the success rate (fraction of runs whose
best_energy_found matches the reference within --success-tolerance), the average
elapsed time, and the standard TTS_p time-to-solution (Ronnow et al. 2014):

    TTS_p = mean_run_time * log(1 - p) / log(1 - success_rate)

p defaults to 0.99. success_rate == 1 is treated as needing a single run
(TTS_p = mean_run_time); == 0 is treated as never succeeding (TTS_p = inf).

Usage:
    python3 summarize_alternative_results.py
Writes GA_alternative_summary.csv and PA_alternative_summary.csv next to this script.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ALGORITHMS = ("GA_alternative", "PA_alternative")

# Names of the leading run-parameter columns in each algorithm's results_*.txt
# files, in file order (the last of which is always the annealing schedule).
PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "GA_alternative": (
        "global_steps_per_temperature",
        "MCS_per_global_steps",
        "number_of_temperatures",
        "schedule",
    ),
    "PA_alternative": (
        "MCS_per_temperature",
        "number_of_temperatures",
        "schedule",
    ),
}

DEFAULT_TTS_TARGET_PROBABILITY = 0.99
DEFAULT_SUCCESS_TOLERANCE = 1e-6

RESULTS_FILE_RE = re.compile(r"results_(\d+)\.txt$")


@dataclass(frozen=True)
class Run:
    seed: int
    params: tuple[str, ...]
    best_energy_found: float
    final_energy: float
    elapsed_time: float


@dataclass(frozen=True)
class GroupSummary:
    seed: int
    params: tuple[str, ...]
    num_runs: int
    best_energy: float
    success_rate: float
    average_time: float
    tts: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize GA_alternative/PA_alternative results into per-(seed, "
            "parameter set) success rate, average time, and TTS."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR,
        help=(
            "directory containing best_energies.txt and the algorithm "
            f"subfolders (default: {SCRIPT_DIR})"
        ),
    )

    parser.add_argument(
        "--success-tolerance",
        type=float,
        default=DEFAULT_SUCCESS_TOLERANCE,
        help=(
            "energy tolerance for counting a run as matching the reference "
            f"best energy (default: {DEFAULT_SUCCESS_TOLERANCE})"
        ),
    )

    parser.add_argument(
        "--tts-target-probability",
        type=float,
        default=DEFAULT_TTS_TARGET_PROBABILITY,
        help=(
            "target confidence p for TTS_p (default: "
            f"{DEFAULT_TTS_TARGET_PROBABILITY}, the standard convention)"
        ),
    )

    args = parser.parse_args()

    if not 0.0 < args.tts_target_probability < 1.0:
        parser.error("--tts-target-probability must be strictly between 0 and 1")

    return args


def load_best_energies(path: Path) -> dict[int, float]:
    best_energies: dict[int, float] = {}

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected 2 columns, got {len(parts)}"
                )

            seed_str, energy_str = parts
            best_energies[int(seed_str)] = float(energy_str)

    return best_energies


def coerce_param(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def parse_algorithm_dir(path: Path) -> tuple[list[Run], list[str]]:
    runs: list[Run] = []
    warnings: list[str] = []

    for results_path in sorted(path.glob("results_*.txt")):
        match = RESULTS_FILE_RE.match(results_path.name)

        if match is None:
            warnings.append(f"{results_path}: unrecognized filename; skipped")
            continue

        seed = int(match.group(1))

        with results_path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 4:
                    warnings.append(
                        f"{results_path}:{line_number}: expected at least 4 "
                        f"columns, got {len(parts)}; skipped"
                    )
                    continue

                params = tuple(parts[:-3])
                best_str, final_str, time_str = parts[-3:]

                try:
                    run = Run(
                        seed=seed,
                        params=params,
                        best_energy_found=float(best_str),
                        final_energy=float(final_str),
                        elapsed_time=float(time_str),
                    )
                except ValueError as exc:
                    warnings.append(f"{results_path}:{line_number}: {exc}; skipped")
                    continue

                runs.append(run)

    return runs, warnings


def compute_tts(
    mean_time: float,
    success_rate: float,
    target_probability: float,
) -> float:
    """Standard percentile time-to-solution, TTS_p (Ronnow et al. 2014)."""
    if success_rate <= 0.0:
        return float("inf")
    if success_rate >= 1.0:
        return mean_time
    restarts_needed = math.log1p(-target_probability) / math.log1p(-success_rate)
    return mean_time * restarts_needed


def summarize(
    runs: list[Run],
    best_energies: dict[int, float],
    success_tolerance: float,
    tts_target_probability: float,
) -> list[GroupSummary]:
    grouped: dict[tuple[int, tuple[str, ...]], list[Run]] = defaultdict(list)

    for run in runs:
        grouped[(run.seed, run.params)].append(run)

    summaries: list[GroupSummary] = []

    for (seed, params), group_runs in grouped.items():
        if seed not in best_energies:
            raise ValueError(f"seed {seed} has no entry in best_energies.txt")

        reference_energy = best_energies[seed]
        times = [run.elapsed_time for run in group_runs]

        successes = sum(
            run.best_energy_found <= reference_energy + success_tolerance
            for run in group_runs
        )
        success_rate = successes / len(group_runs)
        average_time = statistics.mean(times)
        tts = compute_tts(average_time, success_rate, tts_target_probability)
        best_energy = min(run.best_energy_found for run in group_runs)

        summaries.append(
            GroupSummary(
                seed=seed,
                params=params,
                num_runs=len(group_runs),
                best_energy=best_energy,
                success_rate=success_rate,
                average_time=average_time,
                tts=tts,
            )
        )

    def param_sort_key(params: tuple[str, ...]) -> tuple[tuple[int, object], ...]:
        return tuple(
            (0, coerce_param(p)) if isinstance(coerce_param(p), int) else (1, p)
            for p in params
        )

    summaries.sort(key=lambda s: (s.seed, param_sort_key(s.params)))

    return summaries


def write_summary_csv(
    path: Path,
    summaries: list[GroupSummary],
    param_names: tuple[str, ...],
    tts_target_probability: float,
) -> None:
    tts_column = f"tts_p{tts_target_probability:g}"
    fieldnames = [
        "seed",
        *param_names,
        "num_runs",
        "best_energy",
        "success_rate",
        "average_time",
        tts_column,
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        for summary in summaries:
            row = {"seed": summary.seed}
            for name, param in zip(param_names, summary.params):
                row[name] = coerce_param(param)
            row["num_runs"] = summary.num_runs
            row["best_energy"] = f"{summary.best_energy:.12g}"
            row["success_rate"] = f"{summary.success_rate:.6g}"
            row["average_time"] = f"{summary.average_time:.12g}"
            row[tts_column] = (
                f"{summary.tts:.12g}" if math.isfinite(summary.tts) else "inf"
            )
            writer.writerow(row)


def process_algorithm(
    algorithm_dir: Path,
    param_names: tuple[str, ...],
    best_energies: dict[int, float],
    success_tolerance: float,
    tts_target_probability: float,
) -> tuple[list[GroupSummary], list[str]]:
    runs, warnings = parse_algorithm_dir(algorithm_dir)

    if not runs:
        return [], warnings

    for run in runs:
        if len(run.params) != len(param_names):
            raise ValueError(
                f"{algorithm_dir}: expected {len(param_names)} parameter "
                f"columns {param_names}, seed {run.seed} has {len(run.params)}"
            )

    summaries = summarize(
        runs, best_energies, success_tolerance, tts_target_probability
    )

    return summaries, warnings


def main() -> int:
    args = parse_args()

    best_energies_path = args.data_dir / "best_energies.txt"

    if not best_energies_path.exists():
        print(f"error: best energies file not found: {best_energies_path}", file=sys.stderr)
        return 1

    best_energies = load_best_energies(best_energies_path)

    exit_code = 0

    for algorithm in ALGORITHMS:
        algorithm_dir = args.data_dir / algorithm
        param_names = PARAM_NAMES[algorithm]

        if not algorithm_dir.is_dir():
            print(f"error: algorithm directory not found: {algorithm_dir}", file=sys.stderr)
            exit_code = 1
            continue

        summaries, warnings = process_algorithm(
            algorithm_dir,
            param_names,
            best_energies,
            args.success_tolerance,
            args.tts_target_probability,
        )

        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)

        if not summaries:
            print(f"error: no valid runs found under {algorithm_dir}", file=sys.stderr)
            exit_code = 1
            continue

        output_path = args.data_dir / f"{algorithm}_summary.csv"
        write_summary_csv(
            output_path, summaries, param_names, args.tts_target_probability
        )

        print(f"wrote {len(summaries)} groups to {output_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
