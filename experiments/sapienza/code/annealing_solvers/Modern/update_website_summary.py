#!/usr/bin/env python3
"""Write PA / SA / GA num_temps scans into a website summary.csv (SK or EA3D).

Usage (see the bottom of this docstring for the commands of 2026-11-26):
    python update_website_summary.py --summary SUMMARY_CSV \
        [--pa DIR...] [--sa DIR...] [--ga DIR...] [--reference-dirs DIR...] \
        [--reference N=FILE ...] [--drop PROGRAM ...] [--num-temps ALG:N:NT ...] \
        [--success-tolerance TOL] [--snap-to-reference] [--dry-run]

Unlike to_website_summary.py (one num_temps per folder, kept for the
2026-10-08 summary), the folders here may hold several num_temps per N. For
every (algorithm, N) the script keeps the num_temps that minimizes the
median-instance TTS99 (one num_temps for all instances of that N, no
per-instance tuning). Only num_temps run on every instance of that N with the
full number of repeats are candidates. If the median TTS99 is infinite at
every num_temps, the num_temps with the most solved instances (then the
highest mean success, then the largest num_temps) is kept. --num-temps
overrides the choice for one (algorithm, N).

With --per-instance, each instance instead keeps the num_temps with its
lowest TTS99 among those run on it with the full number of repeats (as the
pre-2026-11 EA3D tables did); an instance never solved keeps the num_temps
with its lowest energy (then the largest num_temps).

Reference energy of an instance (energy per spin):
  * with --reference N=FILE (a CSV with columns seed,energy, or a
    two-column whitespace file seed energy): the energy in that file;
  * otherwise: the lowest energy found for it by any run in --pa/--sa/--ga
    and --reference-dirs (all num_temps, anneal best or T = 0 quench) and by
    any row already in summary.csv (including rows removed by --drop).
The energy of a run is min(anneal best, quench energy); the run succeeds if
it is within --success-tolerance of the reference. TTS99 = mean_run_time *
log(0.01) / log(1 - p), mean_run_time if p = 1, inf if p = 0; the run time is
total_time + quench_time (thermalization + MADE training for GA + anneal +
quench).

Rows of summary.csv for an (algorithm, N) present in the new data are
replaced; rows of other (algorithm, N) are kept, and rows of --drop programs
are removed. As in to_website_summary.py, kept rows whose min_energy is above
the reference get success_probability 0 and TTS inf.

--snap-to-reference (for EA3D, whose reference files have 5 decimals while
runs print 6): a new row whose best energy is within the tolerance of the
reference reports the reference energy as min_energy, so that the website
matches it to the verified ground state (generate_results_tables.py compares
energies at 1e-6).

Commands used on 2026-11-26 (run from this folder):
  SK:   --summary ../../../results/sk/summary.csv
        --pa results_parts_pa_mc1_num_temps_N400-N2000_Tend0.1_100inst_10rep_quench
             results_parts_pa_mc1_num_temps_N200_N300_Tend0.1_100inst_10rep_quench
        --sa results_parts_sa_num_temps_N400-N2000_Tend0.1_100inst_10rep_quench
        --reference-dirs <every other SK results_parts_* folder>
        --drop "Global Annealing" --success-tolerance 1.5e-6 --per-instance
  EA3D: --summary ../../../results/ea3d/summary.csv
        --pa results_parts_ea3d_pa_* --sa results_parts_ea3d_sa_* --ga results_parts_ea3d_ga_*
        --reference 1000=../../../../../instances/verified_gs/ea3d/N1000/energies.csv
                    2744=../../../previous_data/multiple_instances_L14/best_energies.txt
        --success-tolerance 5.5e-6 --snap-to-reference --per-instance
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

HARDWARE = "NVIDIA Tesla V100-SXM2-32G"
PROGRAM_NAMES = {"PA": "Population Annealing", "SA": "Simulated Annealing",
                 "GA": "Global Annealing"}
# PA and SA part files have no header line (15 columns in the oldest runs).
COLUMNS = ("repeat schedule t_start t_end num_temps pop_size num_steps_mc thermalization_steps "
           "reweight_mode N instance_seed run_seed final best elapsed therm total qmin qtime "
           "qsweeps").split()
GA_COLUMNS = {"best_min_energy_perspin": "best", "quench_min_energy_perspin": "qmin",
              "total_time": "total", "quench_time": "qtime"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, required=True,
                        help="website summary.csv, updated in place")
    for alg in ("pa", "sa", "ga"):
        parser.add_argument(f"--{alg}", nargs="+", default=[],
                            help=f"{PROGRAM_NAMES[alg.upper()]} result part folders")
    parser.add_argument("--reference-dirs", nargs="+", default=[],
                        help="extra part folders used only for the reference energy")
    parser.add_argument("--reference", nargs="+", default=[], metavar="N=FILE",
                        help="reference energies for size N (CSV seed,energy or 'seed energy')")
    parser.add_argument("--drop", nargs="+", default=[],
                        help="program_name values whose rows are removed")
    parser.add_argument("--num-temps", nargs="+", default=[], metavar="ALG:N:NT",
                        help="force num_temps NT for algorithm ALG (PA/SA/GA) at size N")
    parser.add_argument("--success-tolerance", type=float, default=1.5e-6)
    parser.add_argument("--snap-to-reference", action="store_true")
    parser.add_argument("--per-instance", action="store_true",
                        help="choose num_temps per instance (lowest TTS99) instead of per N")
    parser.add_argument("--dry-run", action="store_true", help="print the tables, write nothing")
    return parser.parse_args()


def read_part(path: str) -> pd.DataFrame:
    with open(path) as handle:
        first = handle.readline().split()
    if not first:
        return pd.DataFrame()
    if first[0] == "repeat":
        frame = pd.read_csv(path, sep=r"\s+").rename(columns=GA_COLUMNS)
        frame = frame[frame["repeat"].astype(str) != "repeat"]
    else:
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS[:len(first)])
    for column in ("N", "instance_seed", "best", "qmin", "total", "qtime", "num_temps",
                   "num_steps_mc", "swap_step"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "qmin" not in frame:
        frame["qmin"] = np.nan
    return frame


def load(folders: list[str], algorithm: str | None) -> pd.DataFrame:
    files = [f for folder in folders for f in sorted(glob.glob(f"{folder}/results_*.txt"))]
    if folders and not files:
        raise SystemExit(f"no result files in {folders}")
    if not files:
        return pd.DataFrame(columns=["N", "instance_seed", "energy"])
    frame = pd.concat([read_part(f) for f in files], ignore_index=True)
    frame["energy"] = frame[["best", "qmin"]].min(axis=1)
    if algorithm is None:
        return frame[["N", "instance_seed", "energy"]]
    if algorithm == "GA":
        frame["parameters"] = [
            f"global_steps_per_temperature={g}, MCS_per_global_steps={l}, "
            f"number_of_temperatures={nt}, schedule={s}"
            for g, l, nt, s in zip(frame.num_steps_mc, frame.swap_step, frame.num_temps,
                                   frame.schedule)]
    else:
        frame["parameters"] = [
            f"MCS_per_temperature={m}, number_of_temperatures={nt}, schedule={s}"
            for m, nt, s in zip(frame.num_steps_mc, frame.num_temps, frame.schedule)]
    frame["program_name"] = PROGRAM_NAMES[algorithm]
    frame["alg"] = algorithm
    frame["run_time"] = frame.total + frame.qtime
    return frame[["alg", "program_name", "N", "instance_seed", "num_temps", "energy", "run_time",
                  "parameters"]]


def read_reference(spec: str) -> pd.Series:
    size, path = spec.split("=", 1)
    if path.endswith(".csv"):
        table = pd.read_csv(path)
    else:
        table = pd.read_csv(path, sep=r"\s+", header=None, names=["seed", "energy"])
    return pd.Series(table.energy.values, index=pd.MultiIndex.from_arrays(
        [np.full(len(table), int(size)), table.seed.astype(int)], names=["N", "instance_seed"]),
        name="reference")


def tts(mean_time: float, p: float) -> float:
    if p <= 0:
        return math.inf
    if p >= 1:
        return mean_time
    return mean_time * math.log(0.01) / math.log1p(-p)


def median(values) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), 50, method="inverted_cdf"))


def main() -> int:
    args = parse_args()
    runs = pd.concat([load(getattr(args, alg.lower()), alg) for alg in PROGRAM_NAMES
                      if getattr(args, alg.lower())], ignore_index=True)
    extra = load(args.reference_dirs, None)
    key = ["N", "instance_seed"]

    summary = pd.read_csv(args.summary, dtype=str, keep_default_na=False)
    existing = pd.DataFrame({"N": summary.N.astype(int), "instance_seed": summary.seed.astype(int),
                             "energy": summary.min_energy.astype(float)})
    found = pd.concat([f[key + ["energy"]] for f in (runs, extra, existing) if len(f)]) \
        .groupby(key).energy.min().rename("reference")
    if args.reference:
        given = pd.concat([read_reference(spec) for spec in args.reference])
        below = found.to_frame().join(given.rename("given"), how="inner")
        below = below[below.reference < below.given - args.success_tolerance]
        if len(below):
            print(f"WARNING: {len(below)} instances have runs below the given reference:\n{below}")
        reference = given
    else:
        reference = found

    runs = runs.join(reference, on=key)
    if runs.reference.isna().any():
        raise SystemExit("instances without a reference energy: "
                         f"{runs[runs.reference.isna()][key].drop_duplicates().values.tolist()}")
    runs["success"] = runs.energy <= runs.reference + args.success_tolerance
    per = runs.groupby(["alg", "program_name", "N", "num_temps", "instance_seed"]).agg(
        min_energy=("energy", "min"), reference=("reference", "first"),
        average_time=("run_time", "mean"), success_probability=("success", "mean"),
        parameters=("parameters", "first"), runs=("success", "size")).reset_index()
    per["TTS"] = [tts(t, p) for t, p in zip(per.average_time, per.success_probability)]

    forced = {}
    for spec in args.num_temps:
        alg, size, nt = spec.split(":")
        forced[(alg.upper(), int(size))] = int(nt)

    chosen = []
    print("alg      N    nt  inst runs  mean p  solved  p=1  median TTS99   q90 TTS99   s/run")
    for (alg, size), group in per.groupby(["alg", "N"]):
        n_inst = group.instance_seed.nunique()
        n_runs = group.runs.max()
        stats = []
        for nt, g in group.groupby("num_temps"):
            complete = len(g) == n_inst and (g.runs == n_runs).all()
            stats.append(dict(nt=nt, complete=complete, inst=len(g), runs=f"{g.runs.min()}",
                              p=g.success_probability.mean(), solved=int((g.success_probability > 0).sum()),
                              full=int((g.success_probability == 1).sum()), med=median(g.TTS),
                              q90=float(np.percentile(g.TTS, 90, method="inverted_cdf")),
                              t=g.average_time.mean()))
        candidates = [s for s in stats if s["complete"]]
        if (alg, size) in forced:
            best = next(s for s in stats if s["nt"] == forced[(alg, size)])
        elif any(math.isfinite(s["med"]) for s in candidates):
            best = min(candidates, key=lambda s: (s["med"], s["nt"]))
        else:
            best = max(candidates, key=lambda s: (s["solved"], s["p"], s["nt"]))
        for s in stats:
            mark = " <==" if s is best else ("" if s["complete"] else "  (incomplete)")
            print(f"{alg:3s} {size:6d} {s['nt']:5d} {s['inst']:5d} {s['runs']:>4s}  {s['p']:6.3f}  "
                  f"{s['solved']:6d} {s['full']:4d}  {s['med']:11.4g}  {s['q90']:10.4g}  "
                  f"{s['t']:8.3g}{mark}")
        if not args.per_instance:
            chosen.append(group[group.num_temps == best["nt"]])
            continue
        # Per-instance choice among the num_temps run on that instance with all repeats.
        full = group[group.runs == n_runs]
        solved = full[np.isfinite(full.TTS)].sort_values(["TTS", "num_temps"])
        unsolved = full[~full.instance_seed.isin(solved.instance_seed)] \
            .sort_values(["min_energy", "num_temps"], ascending=[True, False])
        pick = pd.concat([solved.groupby("instance_seed").head(1),
                          unsolved.groupby("instance_seed").head(1)])
        if len(pick) != n_inst:
            raise SystemExit(f"{alg} N={size}: {n_inst - len(pick)} instances lack a full num_temps")
        print(f"    per instance: median TTS99 {median(pick.TTS):.4g}, "
              f"q90 {np.percentile(pick.TTS, 90, method='inverted_cdf'):.4g}, "
              f"unsolved {int(np.isinf(pick.TTS).sum())}, chosen num_temps "
              f"{pick.num_temps.value_counts().sort_index().to_dict()}")
        chosen.append(pick)
    chosen = pd.concat(chosen, ignore_index=True)

    reached = chosen.min_energy <= chosen.reference + args.success_tolerance
    energy = chosen.min_energy.where(~(args.snap_to_reference & reached), chosen.reference)
    new_rows = pd.DataFrame({
        "N": chosen.N.astype(str),
        "seed": chosen.instance_seed.astype(str),
        "min_energy": energy.map(lambda e: f"{e:.6f}"),
        "average_time": chosen.average_time.map(lambda t: f"{t:.10g}"),
        "success_probability": chosen.success_probability.map(lambda p: f"{p:g}"),
        "TTS": chosen.TTS.map(lambda t: f"{t:.10g}"),
        "hardware": HARDWARE,
        "program_name": chosen.program_name,
        "average_steps": "",
        "parameters": chosen.parameters,
    })

    replaced = set(zip(chosen.program_name, chosen.N.astype(str)))
    drop = summary.program_name.isin(args.drop) | pd.Series(
        [(p, n) in replaced for p, n in zip(summary.program_name, summary.N)], index=summary.index)
    print(f"\nremoving {int(drop.sum())} rows: "
          f"{summary[drop].groupby('program_name').size().to_dict()}")
    kept = summary[~drop].copy()
    kept_energy = kept.min_energy.astype(float).values
    kept_ref = pd.MultiIndex.from_arrays([kept.N.astype(int), kept.seed.astype(int)]).map(
        lambda k: reference.get(k, np.nan))
    above = np.asarray(kept_energy > np.asarray(kept_ref, dtype=float) + args.success_tolerance)
    newly = above & (kept.success_probability.values != "0")
    print(f"kept {len(kept)} rows; {int(above.sum())} do not reach the reference "
          f"({int(newly.sum())} of them had success > 0 before)")
    kept.loc[above, "success_probability"] = "0"
    kept.loc[above, "TTS"] = "inf"
    print("kept rows by program and N:")
    print(kept.groupby(["program_name", kept.N.astype(int)]).size().unstack(0).fillna(0)
          .astype(int).to_string())

    # Kept rows stay in their order, new rows are appended (as to_website_summary.py did).
    output = pd.concat([kept, new_rows], ignore_index=True)[summary.columns]
    if args.dry_run:
        print(f"\ndry run: would write {len(new_rows)} new rows, {len(output)} rows in total")
        return 0
    output.to_csv(args.summary, index=False, lineterminator="\n")
    print(f"\nwrote {len(new_rows)} new rows, {len(output)} rows in total, to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
