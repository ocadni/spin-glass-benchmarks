#!/usr/bin/env python3
"""Summarize population annealing results from DATA_DIR/results_pa.txt.

Usage:
    python analyzer.py DATA_DIR

DATA_DIR is a dataset directory made by merge_results.sh (e.g. analysis/N1200).
Both CSVs described below are written into it, next to results_pa.txt.

Runs are grouped by (N, instance_seed, schedule, t_start); each group becomes
one row with:

    min_energy           lowest energy per spin found by the group's runs
    reference_energy     lowest energy per spin found for that instance by ANY
                         run in the file (all schedules and starting
                         temperatures together)
    success_probability  fraction of the group's runs that reach
                         reference_energy
    TTS                  time to solution

TTS follows the standard time-to-solution convention (Ronnow et al. 2014):
TTS_p = mean_run_time * log(1 - p) / log(1 - success_probability), the total
time needed, running independent restarts, to be p-confident of having found
the reference energy at least once. p defaults to 0.99. success_probability
== 1 is treated as needing a single run (TTS_p = mean_run_time); == 0 is
treated as never succeeding (TTS_p = inf).

The rows are written to a CSV file.

A second CSV ranks the (schedule, t_start) combinations: for each N it gives
the percentage of instances on which each combination is the best one, i.e.
has the highest success_probability (fraction of runs reaching the reference
energy, the best-known ground state). TTS is not used for this ranking: all
runs take about the same time, so it would only add timing noise. If several
combinations tie for the highest success_probability on an instance, each of
them is counted as best on it, so the percentages can sum to more than 100.
This table is also printed.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


INPUT_NAME = "results_pa.txt"
OUTPUT_NAME = "results_pa_summary.csv"
BEST_OUTPUT_NAME = "results_pa_best_params.csv"
DEFAULT_TTS_TARGET_PROBABILITY = 0.99

ENERGY_COLUMNS = ("best_min_energy_perspin", "final_min_energy_perspin")

# Columns that are expected to be constant inside a group; a warning is
# printed if they are not, since the group would then mix different setups.
CONFIG_COLUMNS = (
    "t_end", "num_temps", "pop_size", "num_steps_mc",
    "thermalization_steps", "reweight_mode",
)


@dataclass(frozen=True)
class Run:
    n: int
    instance_seed: int
    schedule: str
    t_start: float
    run_seed: int
    energy_per_spin: float
    elapsed_time: float
    config: tuple[str, ...]


@dataclass(frozen=True)
class GroupSummary:
    n: int
    instance_seed: int
    schedule: str
    t_start: float
    runs: int
    min_energy: float
    reference_energy: float
    successes: int
    success_probability: float
    mean_elapsed_time: float
    tts: float


@dataclass(frozen=True)
class CombinationWins:
    n: int
    schedule: str
    t_start: float
    instances: int
    wins: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize population annealing results per instance, schedule and t_start."
    )

    parser.add_argument(
        "data_dir",
        type=Path,
        help="dataset directory made by merge_results.sh, e.g. analysis/N1200",
    )

    parser.add_argument(
        "--input",
        type=Path,
        help=f"input results file (default: DATA_DIR/{INPUT_NAME})",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help=f"output summary CSV, overwritten on each run (default: DATA_DIR/{OUTPUT_NAME})",
    )

    parser.add_argument(
        "--best-output",
        type=Path,
        help=(
            "output CSV with the percentage of instances on which each "
            "(schedule, t_start) has the highest success probability, "
            f"overwritten on each run (default: DATA_DIR/{BEST_OUTPUT_NAME})"
        ),
    )

    parser.add_argument(
        "--energy-column",
        choices=ENERGY_COLUMNS,
        default=ENERGY_COLUMNS[0],
        help=(
            "which energy to use: the best seen during the anneal, or the one "
            f"at the end of it (default: {ENERGY_COLUMNS[0]})"
        ),
    )

    parser.add_argument(
        "--success-tolerance",
        type=float,
        default=5e-7,
        help=(
            "energy tolerance for counting a run as reaching the reference "
            "energy (energies are rounded to 6 decimals, so the default "
            "counts equal printed values as ties)"
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

    args.input = args.input or args.data_dir / INPUT_NAME
    args.output = args.output or args.data_dir / OUTPUT_NAME
    args.best_output = args.best_output or args.data_dir / BEST_OUTPUT_NAME

    return args


def parse_results(path: Path, energy_column: str) -> list[Run]:
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")

    runs: list[Run] = []

    with path.open(encoding="utf-8") as handle:
        header = handle.readline().split()
        required = {
            "schedule", "t_start", "N", "instance_seed", "run_seed",
            "elapsed_time", energy_column, *CONFIG_COLUMNS,
        }
        missing = required.difference(header)
        if missing:
            raise ValueError(f"{path}: header is missing columns {sorted(missing)}")

        for line_number, raw_line in enumerate(handle, start=2):
            parts = raw_line.split()

            if not parts:
                continue

            # merge_results.sh writes a single header, but skip repeated ones
            # in case part files were concatenated by hand.
            if parts == header:
                continue

            if len(parts) != len(header):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(header)} columns, got {len(parts)}"
                )

            row = dict(zip(header, parts))
            runs.append(
                Run(
                    n=int(row["N"]),
                    instance_seed=int(row["instance_seed"]),
                    schedule=row["schedule"],
                    t_start=float(row["t_start"]),
                    run_seed=int(row["run_seed"]),
                    energy_per_spin=float(row[energy_column]),
                    elapsed_time=float(row["elapsed_time"]),
                    config=tuple(row[column] for column in CONFIG_COLUMNS),
                )
            )

    return runs


def compute_tts(
    mean_time: float,
    success_probability: float,
    target_probability: float,
) -> float:
    """Standard percentile time-to-solution, TTS_p (Ronnow et al. 2014)."""
    if success_probability <= 0.0:
        return float("inf")
    if success_probability >= 1.0:
        return mean_time
    restarts_needed = math.log1p(-target_probability) / math.log1p(-success_probability)
    return mean_time * restarts_needed


def summarize_groups(
    runs: list[Run],
    success_tolerance: float,
    tts_target_probability: float,
) -> tuple[list[GroupSummary], list[str]]:
    reference_energies: dict[tuple[int, int], float] = {}
    grouped: dict[tuple[int, int, str, float], list[Run]] = defaultdict(list)
    schedule_order: dict[str, int] = {}

    for run in runs:
        instance = (run.n, run.instance_seed)
        reference_energies[instance] = min(
            reference_energies.get(instance, math.inf), run.energy_per_spin
        )
        grouped[(run.n, run.instance_seed, run.schedule, run.t_start)].append(run)
        schedule_order.setdefault(run.schedule, len(schedule_order))

    summaries: list[GroupSummary] = []
    warnings: list[str] = []

    # Keep schedules in the order they appear in the file rather than
    # alphabetically, so the table follows the order the runs were launched.
    def sort_key(key: tuple[int, int, str, float]) -> tuple[int, int, int, float]:
        n, instance_seed, schedule, t_start = key
        return n, instance_seed, schedule_order[schedule], t_start

    for key in sorted(grouped, key=sort_key):
        n, instance_seed, schedule, t_start = key
        group_runs = grouped[key]

        if len({run.config for run in group_runs}) > 1:
            warnings.append(
                f"N={n} instance_seed={instance_seed} schedule={schedule} "
                f"t_start={t_start:g} mixes runs with different {', '.join(CONFIG_COLUMNS)}"
            )

        reference_energy = reference_energies[(n, instance_seed)]
        successes = sum(
            run.energy_per_spin <= reference_energy + success_tolerance
            for run in group_runs
        )
        success_probability = successes / len(group_runs)
        mean_elapsed_time = statistics.mean(run.elapsed_time for run in group_runs)

        summaries.append(
            GroupSummary(
                n=n,
                instance_seed=instance_seed,
                schedule=schedule,
                t_start=t_start,
                runs=len(group_runs),
                min_energy=min(run.energy_per_spin for run in group_runs),
                reference_energy=reference_energy,
                successes=successes,
                success_probability=success_probability,
                mean_elapsed_time=mean_elapsed_time,
                tts=compute_tts(
                    mean_elapsed_time, success_probability, tts_target_probability
                ),
            )
        )

    return summaries, warnings


def count_best_success(summaries: list[GroupSummary]) -> list[CombinationWins]:
    by_instance: dict[tuple[int, int], list[GroupSummary]] = defaultdict(list)
    for summary in summaries:
        by_instance[(summary.n, summary.instance_seed)].append(summary)

    # Insertion order follows the (already sorted) summaries, so combinations
    # come out in the same schedule / t_start order as the summary CSV.
    wins: dict[tuple[int, str, float], int] = {}
    instances_per_n: dict[int, int] = defaultdict(int)

    for summary in summaries:
        wins.setdefault((summary.n, summary.schedule, summary.t_start), 0)

    for (n, _), instance_summaries in by_instance.items():
        instances_per_n[n] += 1
        best_probability = max(
            summary.success_probability for summary in instance_summaries
        )
        for summary in instance_summaries:
            if summary.success_probability == best_probability:
                wins[(n, summary.schedule, summary.t_start)] += 1

    return [
        CombinationWins(
            n=n,
            schedule=schedule,
            t_start=t_start,
            instances=instances_per_n[n],
            wins=win_count,
        )
        for (n, schedule, t_start), win_count in sorted(
            wins.items(), key=lambda item: item[0][0]
        )
    ]


FIELDNAMES = [
    "N", "instance_seed", "schedule", "t_start", "runs", "min_energy",
    "reference_energy", "successes", "success_probability", "average_time", "TTS",
]

BEST_FIELDNAMES = [
    "N", "schedule", "t_start", "instances", "best_count", "best_percentage",
]


def summary_row(summary: GroupSummary) -> dict[str, str]:
    return {
        "N": str(summary.n),
        "instance_seed": str(summary.instance_seed),
        "schedule": summary.schedule,
        "t_start": f"{summary.t_start:g}",
        "runs": str(summary.runs),
        "min_energy": f"{summary.min_energy:.6f}",
        "reference_energy": f"{summary.reference_energy:.6f}",
        "successes": str(summary.successes),
        "success_probability": f"{summary.success_probability:.4g}",
        "average_time": f"{summary.mean_elapsed_time:.6g}",
        "TTS": f"{summary.tts:.6g}",
    }


def best_row(combination: CombinationWins) -> dict[str, str]:
    return {
        "N": str(combination.n),
        "schedule": combination.schedule,
        "t_start": f"{combination.t_start:g}",
        "instances": str(combination.instances),
        "best_count": str(combination.wins),
        "best_percentage": f"{100.0 * combination.wins / combination.instances:.2f}",
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_table(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    widths = {
        name: max(len(name), *(len(row[name]) for row in rows))
        for name in fieldnames
    }

    print("  ".join(name.rjust(widths[name]) for name in fieldnames))
    for row in rows:
        print("  ".join(row[name].rjust(widths[name]) for name in fieldnames))


def main() -> int:
    args = parse_args()

    runs = parse_results(args.input, args.energy_column)

    if not runs:
        print(f"error: no runs found in {args.input}", file=sys.stderr)
        return 1

    summaries, warnings = summarize_groups(
        runs,
        args.success_tolerance,
        args.tts_target_probability,
    )
    combinations = count_best_success(summaries)

    write_csv(args.output, FIELDNAMES, [summary_row(summary) for summary in summaries])

    best_rows = [best_row(combination) for combination in combinations]
    write_csv(args.best_output, BEST_FIELDNAMES, best_rows)
    print_table(BEST_FIELDNAMES, best_rows)

    print()
    print(f"parsed {len(runs)} runs from {args.input}")
    print(f"wrote {len(summaries)} rows to {args.output}")
    print(f"wrote {len(combinations)} rows to {args.best_output}")

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
