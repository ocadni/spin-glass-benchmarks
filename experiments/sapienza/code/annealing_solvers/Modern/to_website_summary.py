#!/usr/bin/env python3
"""Write the PA, SA and GA SK results into the website summary.csv.

Usage:
    python to_website_summary.py --pa PA_PARTS_DIR... --sa SA_PARTS_DIR... \
        --ga GA_PARTS_DIR... [--summary SUMMARY_CSV]

Reads the result part files written by run_pa_sk_dariah_batch.pbs (PA and SA,
no header) and run_ga_sk_dariah.pbs (GA, with a header line), and writes one
row per (algorithm, N, instance seed) into experiments/sapienza/results/sk/
summary.csv, the file scripts/generate_results_tables.py reads.

The reference energy of an instance is the lowest energy per spin found for it
by any algorithm: any PA, SA or GA run (anneal best or T = 0 quench) and any
row already in summary.csv (the greedy solvers). The energy of a run is the
lower of its anneal best and its quench energy, and the run succeeds if that
energy is within --success-tolerance of the reference. TTS follows the
standard convention (Ronnow et al. 2014): TTS_p = mean_run_time *
log(1 - p) / log(1 - success_probability), mean_run_time when every run
succeeds and inf when none does. The run time is total_time + quench_time
(for GA this includes the MADE training).

The rows already in summary.csv measured success against their own
algorithm's best energy. Rows whose min_energy is above the common reference
are rewritten to success_probability = 0 and TTS = inf; the others already
reach the reference, so their success probability is unchanged. Earlier PA,
SA and GA rows are replaced, so the script can be re-run.
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SUMMARY = REPO_ROOT / "experiments" / "sapienza" / "results" / "sk" / "summary.csv"
HARDWARE = "NVIDIA Tesla V100-SXM2-32G"
PROGRAM_NAMES = {"PA": "Population Annealing", "SA": "Simulated Annealing",
                 "GA": "Global Annealing"}
# PA and SA part files have no header line.
COLUMNS = ("repeat schedule t_start t_end num_temps pop_size num_steps_mc thermalization_steps "
           "reweight_mode N instance_seed run_seed final best elapsed therm total qmin qtime "
           "qsweeps").split()
GA_COLUMNS = {"best_min_energy_perspin": "best", "quench_min_energy_perspin": "qmin",
              "total_time": "total", "quench_time": "qtime"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pa", nargs="+", required=True, help="PA result part folders")
    parser.add_argument("--sa", nargs="+", required=True, help="SA result part folders")
    parser.add_argument("--ga", nargs="+", required=True, help="GA result part folders")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                        help=f"website summary.csv, updated in place (default: {DEFAULT_SUMMARY})")
    parser.add_argument("--success-tolerance", type=float, default=1.5e-6,
                        help="energy per spin tolerance for success (default: 1.5e-6)")
    parser.add_argument("--tts-target-probability", type=float, default=0.99)
    return parser.parse_args()


def load(folders: list[str], algorithm: str) -> pd.DataFrame:
    files = [f for folder in folders for f in sorted(glob.glob(f"{folder}/results_*_*.txt"))]
    if not files:
        raise SystemExit(f"no result files in {folders}")
    if algorithm == "GA":
        frame = pd.concat(pd.read_csv(f, sep=r"\s+") for f in files).rename(columns=GA_COLUMNS)
        frame["parameters"] = frame.apply(
            lambda r: f"global_steps_per_temperature={r.num_steps_mc}, "
                      f"MCS_per_global_steps={r.swap_step}, "
                      f"number_of_temperatures={r.num_temps}, schedule={r.schedule}", axis=1)
    else:
        frame = pd.concat(pd.read_csv(f, sep=r"\s+", header=None, names=COLUMNS) for f in files)
        frame["parameters"] = frame.apply(
            lambda r: f"MCS_per_temperature={r.num_steps_mc}, "
                      f"number_of_temperatures={r.num_temps}, schedule={r.schedule}", axis=1)
    frame["program_name"] = PROGRAM_NAMES[algorithm]
    return frame[["program_name", "N", "instance_seed", "best", "qmin", "total", "qtime",
                  "parameters"]]


def tts(mean_time: float, p: float, target: float) -> float:
    if p <= 0:
        return math.inf
    if p >= 1:
        return mean_time
    return mean_time * math.log1p(-target) / math.log1p(-p)


def main() -> int:
    args = parse_args()
    runs = pd.concat([load(args.pa, "PA"), load(args.sa, "SA"), load(args.ga, "GA")])
    runs["energy"] = runs[["best", "qmin"]].min(axis=1)
    runs["run_time"] = runs.total + runs.qtime

    # Read as strings so the rows that are kept are written back unchanged.
    summary = pd.read_csv(args.summary, dtype=str, keep_default_na=False)
    summary = summary[~summary.program_name.isin(PROGRAM_NAMES.values())]
    if "parameters" not in summary.columns:
        summary["parameters"] = ""
    key = ["N", "instance_seed"]
    existing = pd.DataFrame({"N": summary.N.astype(int), "instance_seed": summary.seed.astype(int),
                             "energy": summary.min_energy.astype(float)})
    reference = (pd.concat([runs[key + ["energy"]], existing]).groupby(key).energy.min()
                 .rename("reference"))

    runs = runs.join(reference, on=key)
    runs["success"] = runs.energy <= runs.reference + args.success_tolerance
    per_instance = runs.groupby(["program_name", "N", "instance_seed"]).agg(
        min_energy=("energy", "min"), average_time=("run_time", "mean"),
        success_probability=("success", "mean"), parameters=("parameters", "first"),
        runs=("success", "size")).reset_index()
    if per_instance.runs.nunique() != 1:
        print(f"warning: runs per instance vary: {sorted(per_instance.runs.unique())}")
    new_rows = pd.DataFrame({
        "N": per_instance.N.astype(str),
        "seed": per_instance.instance_seed.astype(str),
        "min_energy": per_instance.min_energy.map(lambda e: f"{e:.6f}"),
        "average_time": per_instance.average_time.map(lambda t: f"{t:.10g}"),
        "success_probability": per_instance.success_probability.map(lambda p: f"{p:g}"),
        "TTS": [f"{tts(t, p, args.tts_target_probability):.10g}" for t, p in
                zip(per_instance.average_time, per_instance.success_probability)],
        "hardware": HARDWARE,
        "program_name": per_instance.program_name,
        "parameters": per_instance.parameters,
    })

    above = (existing.join(reference, on=key)
             .pipe(lambda d: d.energy > d.reference + args.success_tolerance)).to_numpy()
    summary.loc[above, "success_probability"] = "0"
    summary.loc[above, "TTS"] = "inf"
    print(f"{int(above.sum())} of {len(summary)} existing rows do not reach the reference energy")

    output = pd.concat([summary, new_rows], ignore_index=True)[summary.columns]
    output.to_csv(args.summary, index=False, lineterminator="\n")
    print(f"wrote {len(new_rows)} PA/SA/GA rows to {args.summary}")

    report = per_instance.assign(never=per_instance.success_probability == 0).groupby(
        ["N", "program_name"]).agg(instances=("never", "size"), mean_success=(
            "success_probability", "mean"), never_solved=("never", "sum"))
    print(report.unstack("program_name").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
