#!/usr/bin/env python3
"""Convert greedy results.txt files to the website summary.csv format.

Supported input formats:

New format:
    repeat N instance_seed run_seed min_energy_perspin elapsed_time

Old format:
    N instance_seed run_seed min_energy_perspin elapsed_time

The "repeat" column in the new format is ignored.

The generated rows are appended to summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

DEFAULT_INPUT = SCRIPT_DIR / "results.txt"

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "sapienza"
    / "results"
    / "sk"
    / "summary.csv"
)

DEFAULT_PROGRAM_NAME = "Random Greedy"
MAX_RUN_SEED = 2**31 - 1


@dataclass(frozen=True)
class Run:
    n: int
    instance_seed: int
    run_seed: int
    energy_per_spin: float
    elapsed_time: float


@dataclass(frozen=True)
class InstanceSummary:
    n: int
    instance_seed: int
    runs: int
    best_energy: float
    best_run_seed: int
    best_elapsed_time: float
    mean_energy: float
    std_energy: float
    mean_elapsed_time: float
    success_probability: float
    tts: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append greedy results to the website summary.csv."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"input results file (default: {DEFAULT_INPUT})",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output website summary.csv (default: {DEFAULT_OUTPUT})",
    )

    parser.add_argument(
        "--program-name",
        default=DEFAULT_PROGRAM_NAME,
        help=f"program name to write in summary.csv (default: {DEFAULT_PROGRAM_NAME})",
    )

    parser.add_argument(
        "--hardware",
        default=f"{platform.system()} {platform.machine()}",
        help="hardware label to write in summary.csv",
    )

    parser.add_argument(
        "--success-tolerance",
        type=float,
        default=1e-12,
        help="energy tolerance for counting runs tied with the best energy",
    )

    parser.add_argument(
        "--strict-run-seed",
        action="store_true",
        help=f"skip rows whose run_seed is outside [0, {MAX_RUN_SEED}]",
    )

    return parser.parse_args()


def parse_results(
    path: Path,
    strict_run_seed: bool,
) -> tuple[list[Run], list[str]]:
    runs: list[Run] = []
    warnings: list[str] = []

    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()

            # New-format header:
            # repeat N instance_seed run_seed min_energy_perspin elapsed_time
            if parts == [
                "repeat",
                "N",
                "instance_seed",
                "run_seed",
                "min_energy_perspin",
                "elapsed_time",
            ]:
                continue

            # Old-format header:
            # N instance_seed run_seed min_energy_perspin elapsed_time
            if parts == [
                "N",
                "instance_seed",
                "run_seed",
                "min_energy_perspin",
                "elapsed_time",
            ]:
                continue

            # New format:
            # repeat N instance_seed run_seed min_energy_perspin elapsed_time
            if len(parts) == 6:
                (
                    _repeat_str,
                    n_str,
                    instance_seed_str,
                    run_seed_str,
                    energy_str,
                    time_str,
                ) = parts

            # Old format:
            # N instance_seed run_seed min_energy_perspin elapsed_time
            elif len(parts) == 5:
                (
                    n_str,
                    instance_seed_str,
                    run_seed_str,
                    energy_str,
                    time_str,
                ) = parts

            else:
                warnings.append(
                    f"{path}:{line_number}: expected 5 or 6 columns, "
                    f"got {len(parts)}; skipped"
                )
                continue

            try:
                run = Run(
                    n=int(n_str),
                    instance_seed=int(instance_seed_str),
                    run_seed=int(run_seed_str),
                    energy_per_spin=float(energy_str),
                    elapsed_time=float(time_str),
                )

            except ValueError as exc:
                warnings.append(
                    f"{path}:{line_number}: {exc}; skipped"
                )
                continue

            if run.run_seed < 0 or run.run_seed > MAX_RUN_SEED:
                message = (
                    f"{path}:{line_number}: run_seed {run.run_seed} "
                    f"is outside [0, {MAX_RUN_SEED}]"
                )

                if strict_run_seed:
                    warnings.append(message + "; skipped")
                    continue

                warnings.append(message)

            runs.append(run)

    return runs, warnings


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    return statistics.stdev(values)


def summarize_instances(
    runs: list[Run],
    success_tolerance: float,
) -> list[InstanceSummary]:
    grouped: dict[tuple[int, int], list[Run]] = defaultdict(list)

    for run in runs:
        grouped[(run.n, run.instance_seed)].append(run)

    summaries: list[InstanceSummary] = []

    for (n, instance_seed), instance_runs in sorted(grouped.items()):
        best = min(
            instance_runs,
            key=lambda run: (
                run.energy_per_spin,
                run.elapsed_time,
                run.run_seed,
            ),
        )

        energies = [
            run.energy_per_spin
            for run in instance_runs
        ]

        times = [
            run.elapsed_time
            for run in instance_runs
        ]

        successes = sum(
            energy <= best.energy_per_spin + success_tolerance
            for energy in energies
        )

        success_probability = successes / len(instance_runs)

        tts = statistics.mean(times) / success_probability

        summaries.append(
            InstanceSummary(
                n=n,
                instance_seed=instance_seed,
                runs=len(instance_runs),
                best_energy=best.energy_per_spin,
                best_run_seed=best.run_seed,
                best_elapsed_time=best.elapsed_time,
                mean_energy=statistics.mean(energies),
                std_energy=stddev(energies),
                mean_elapsed_time=statistics.mean(times),
                success_probability=success_probability,
                tts=tts,
            )
        )

    return summaries


def write_summary_csv(
    path: Path,
    instance_summaries: list[InstanceSummary],
    hardware: str,
    program_name: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write the header only if the file does not exist or is empty.
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "N",
                "seed",
                "min_energy",
                "average_time",
                "success_probability",
                "TTS",
                "hardware",
                "program_name",
            ],
            lineterminator="\n",
        )

        if write_header:
            writer.writeheader()

        for summary in instance_summaries:
            writer.writerow(
                {
                    "N": summary.n,
                    "seed": summary.instance_seed,
                    "min_energy": f"{summary.best_energy:.12g}",
                    "average_time": f"{summary.mean_elapsed_time:.12g}",
                    "success_probability": f"{summary.success_probability:.12g}",
                    "TTS": f"{summary.tts:.12g}",
                    "hardware": hardware,
                    "program_name": program_name,
                }
            )


def main() -> int:
    args = parse_args()

    runs, warnings = parse_results(
        args.input,
        args.strict_run_seed,
    )

    if not runs:
        print(
            f"error: no valid runs found in {args.input}",
            file=sys.stderr,
        )
        return 1

    instance_summaries = summarize_instances(
        runs,
        args.success_tolerance,
    )

    write_summary_csv(
        args.output,
        instance_summaries,
        args.hardware,
        args.program_name,
    )

    print(f"appended results to {args.output}")
    print(f"parsed {len(runs)} runs")
    print(f"appended {len(instance_summaries)} instance summaries")

    if warnings:
        for warning in warnings:
            print(
                f"warning: {warning}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())