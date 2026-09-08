#!/usr/bin/env python3
"""Regenerate the per-family result tables in docs/results.qmd.

Scans experiments/<researcher>/results/<family>/summary.csv for rows with
columns (any order): N, seed, min_energy, average_time, success_probability,
TTS, hardware, program_name, plus optional average_steps. Missing step
measurements are displayed as an em dash in the All section's table (the
Leaderboard does not show steps at all).

The family (sk/ea2d/ea3d/rrg) is inferred from the name of the directory a
summary.csv lives in, and the researcher from the path component right after
experiments/. For every (family, N, seed) instance seen across all
summary.csv files, the Leaderboard section reports the single best run:
ranked first by lowest min_energy, energies within 1e-6 of the minimum
treated as tied, ties broken by lowest TTS (and, in the
vanishingly unlikely case both are equal too, by program_name then hardware,
for a fully deterministic result regardless of file/row order). The All
section instead lists every run (with runtime, success probability, and
average steps, none of which the Leaderboard shows), grouped by algorithm
then family, with each researcher's notes.md (if present and non-empty)
shown as a collapsible toggle.

Usage: python scripts/generate_results_tables.py
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
INSTANCES_DIR = REPO_ROOT / "instances"
VERIFIED_GS_DIR = INSTANCES_DIR / "verified_gs"
RESULTS_QMD = REPO_ROOT / "docs" / "results.qmd"

KNOWN_FAMILIES = {"sk", "ea2d", "ea3d", "rrg"}
EXPECTED_FIELDS = {
    "N",
    "seed",
    "min_energy",
    "average_time",
    "success_probability",
    "TTS",
    "hardware",
    "program_name",
}

FAMILY_LABELS = {
    "sk": "Sherrington-Kirkpatrick (SK)",
    "ea2d": "Edwards-Anderson 2D (EA2D)",
    "ea3d": "Edwards-Anderson 3D (EA3D)",
    "rrg": "Random Regular Graph (RRG)",
}

# Families shown in the "All" section's per-algorithm tabs. Keep in sync with
# which "Best" sections are un-hidden (.content-hidden) in docs/results.qmd.
VISIBLE_FAMILIES = ("sk",)

# Algorithm tabs shown in the "All" section, even before every algorithm has
# uploaded results. Extra algorithm names found in summary.csv are appended.
VISIBLE_ALGORITHMS = ("Greedy", "Random", "Reluctant")

# Absolute tolerance in the reported energy units, applied against the
# instance minimum so that selection does not depend on row order.
ENERGY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Row:
    n: int
    seed: int
    min_energy: float
    average_time: float
    success_probability: float
    tts: float
    hardware: str
    program_name: str
    researcher: str
    average_steps: float | None = None


def find_summary_files() -> list[Path]:
    return sorted(EXPERIMENTS_DIR.rglob("summary.csv"))


def infer_family(path: Path) -> str | None:
    family = path.parent.name
    if family not in KNOWN_FAMILIES:
        print(f"warning: skipping {path} (unrecognized family {family!r})", file=sys.stderr)
        return None
    return family


def infer_researcher(path: Path) -> str:
    return path.relative_to(EXPERIMENTS_DIR).parts[0]


def parse_summary_file(path: Path) -> list[Row]:
    rows: list[Row] = []
    researcher = infer_researcher(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not EXPECTED_FIELDS.issubset(reader.fieldnames):
            print(
                f"warning: {path}: expected fields {sorted(EXPECTED_FIELDS)}, got {reader.fieldnames}",
                file=sys.stderr,
            )
            return rows
        for line_num, record in enumerate(reader, start=2):
            try:
                rows.append(
                    Row(
                        n=int(record["N"]),
                        seed=int(record["seed"]),
                        min_energy=float(record["min_energy"]),
                        average_time=float(record["average_time"]),
                        success_probability=float(record["success_probability"]),
                        tts=float(record["TTS"]),
                        hardware=record["hardware"],
                        program_name=record["program_name"],
                        researcher=researcher,
                        average_steps=(
                            float(record["average_steps"])
                            if (record.get("average_steps") or "").strip() else None
                        ),
                    )
                )
            except (ValueError, TypeError) as exc:
                print(f"warning: {path}:{line_num}: {exc}", file=sys.stderr)
    return rows


def collect_rows_by_family() -> dict[str, list[Row]]:
    rows_by_family: dict[str, list[Row]] = {family: [] for family in KNOWN_FAMILIES}
    for summary_path in find_summary_files():
        family = infer_family(summary_path)
        if family is None:
            continue
        rows_by_family[family].extend(parse_summary_file(summary_path))
    return rows_by_family


def load_verified_gs() -> dict[tuple[str, int, int], float]:
    """Map (family, N, seed) -> exact ground-state energy.

    Reads instances/verified_gs/<FAMILY>/N<size>/energies.csv (columns:
    seed, energy), one file per family/size directory, mirroring the
    corresponding instances/<family>/N<size>/ layout. The family directory
    name's casing follows instances/ (e.g. "EA3D"), so matching against the
    lowercase KNOWN_FAMILIES keys is case-insensitive.
    """
    verified: dict[tuple[str, int, int], float] = {}
    if not VERIFIED_GS_DIR.is_dir():
        return verified
    for family_dir in sorted(VERIFIED_GS_DIR.iterdir()):
        if not family_dir.is_dir():
            continue
        family = family_dir.name.lower()
        if family not in KNOWN_FAMILIES:
            print(f"warning: skipping {family_dir} (unrecognized family {family_dir.name!r})", file=sys.stderr)
            continue
        for size_dir in sorted(family_dir.iterdir()):
            match = re.fullmatch(r"N(\d+)", size_dir.name)
            if not match:
                continue
            n = int(match.group(1))
            csv_path = size_dir / "energies.csv"
            if not csv_path.is_file():
                continue
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or not {"seed", "energy"}.issubset(reader.fieldnames):
                    print(
                        f"warning: {csv_path}: expected fields {{'seed', 'energy'}}, got {reader.fieldnames}",
                        file=sys.stderr,
                    )
                    continue
                for line_num, record in enumerate(reader, start=2):
                    try:
                        verified[(family, n, int(record["seed"]))] = float(record["energy"])
                    except (ValueError, TypeError) as exc:
                        print(f"warning: {csv_path}:{line_num}: {exc}", file=sys.stderr)
    return verified


def render_table(family: str, rows: list[Row], verified: dict[tuple[str, int, int], float]) -> str:
    verified_for_family = {
        (n, seed): energy for (f, n, seed), energy in verified.items() if f == family
    }

    if not rows and not verified_for_family:
        return (
            "TODO: populate from experiments.\n\n"
            "| N | Seed | Best Algorithm | Hardware | Energy | Reference Energy | TTS (s) |\n"
            "|---|------|-----------------|----------|--------|-------------------|-----|\n"
            "| TODO | | | | | | |"
        )

    by_instance: dict[tuple[int, int], list[Row]] = {}
    for row in rows:
        by_instance.setdefault((row.n, row.seed), []).append(row)

    best_by_key: dict[tuple[int, int], Row] = {}
    for key, instance_rows in by_instance.items():
        minimum_energy = min(r.min_energy for r in instance_rows)
        tied_rows = [
            r for r in instance_rows
            if r.min_energy - minimum_energy <= ENERGY_TOLERANCE
        ]
        best_by_key[key] = min(tied_rows, key=lambda r: (r.tts, r.program_name, r.hardware))

    keys_by_n: dict[int, list[int]] = {}
    for (n, seed) in set(best_by_key) | set(verified_for_family):
        keys_by_n.setdefault(n, []).append(seed)

    blocks = ["::: {.panel-tabset}"]
    for n in sorted(keys_by_n):
        lines = [
            f"\n### N = {n}\n",
            "| Seed | Best Algorithm | Hardware | Energy | Reference Energy | TTS (s) |",
            "|------|----------------|----------|--------|-------------------|---------|",
        ]
        for seed in sorted(keys_by_n[n]):
            best = best_by_key.get((n, seed))
            reference_energy = verified_for_family.get((n, seed))
            if best is None:
                # Verified ground state with no submitted solver run yet.
                lines.append(f"| {seed}† | — | — | — | {reference_energy:.7g} | — |")
                continue
            if reference_energy is None:
                seed_cell = str(best.seed)
                reference_cell = "—"
                algorithm_cell = best.program_name
            else:
                seed_cell = f"{best.seed}†"
                reference_cell = f"{reference_energy:.7g}"
                matched = abs(best.min_energy - reference_energy) <= ENERGY_TOLERANCE
                algorithm_cell = best.program_name if matched else "—"
            lines.append(
                f"| {seed_cell} "
                f"| {algorithm_cell} | {best.hardware} "
                f"| {best.min_energy:.7g} | {reference_cell} | {best.tts:.7g} |"
            )
        blocks.append("\n".join(lines))
    blocks.append(":::")
    return "\n".join(blocks)


def format_steps(row: Row) -> str:
    return f"{row.average_steps:.7g}" if row.average_steps is not None else "—"


def format_probability(value: float) -> str:
    """Format a probability with at most 5 decimal places, trailing zeros trimmed."""
    text = f"{value:.5f}".rstrip("0").rstrip(".")
    return text if text else "0"


def render_raw_table(rows: list[Row]) -> str:
    rows_by_n: dict[int, list[Row]] = {}
    for row in rows:
        rows_by_n.setdefault(row.n, []).append(row)

    blocks = ["::: {.panel-tabset}"]
    for n in sorted(rows_by_n):
        lines = [
            f"\n#### N = {n}\n",
            "| Seed | Energy | Runtime (s) | Average steps | Success Probability | Hardware |",
            "|------|--------|-------------|---------------|----------------------|----------|",
        ]
        for row in sorted(rows_by_n[n], key=lambda r: (r.seed, r.hardware, r.tts)):
            lines.append(
                f"| {row.seed} | {row.min_energy:.7g} | {row.average_time:.7g} "
                f"| {format_steps(row)} "
                f"| {format_probability(row.success_probability)} | {row.hardware} |"
            )
        blocks.append("\n".join(lines))
    blocks.append(":::")
    return "\n".join(blocks)


def find_notes(researcher: str, family: str) -> Path | None:
    notes_path = EXPERIMENTS_DIR / researcher / "results" / family / "notes.md"
    return notes_path if notes_path.exists() else None


def render_notes_toggles(rows: list[Row], family: str) -> list[str]:
    blocks: list[str] = []
    for researcher in sorted({row.researcher for row in rows}):
        notes_path = find_notes(researcher, family)
        if notes_path is None:
            continue
        content = notes_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        blocks.append(f'\n<details>\n<summary>Notes</summary>\n\n{content}\n\n</details>\n')
    return blocks


def render_all_section(rows_by_family: dict[str, list[Row]]) -> str:
    algorithms: dict[str, dict[str, list[Row]]] = {}
    for family in VISIBLE_FAMILIES:
        for row in rows_by_family.get(family, []):
            algorithms.setdefault(row.program_name, {}).setdefault(family, []).append(row)

    if not algorithms and not VISIBLE_ALGORITHMS:
        return "TODO: populate from experiments."

    blocks = ["::: {.panel-tabset}"]
    extra_algorithms = sorted(
        algorithm for algorithm in algorithms if algorithm not in VISIBLE_ALGORITHMS
    )
    for algorithm in (*VISIBLE_ALGORITHMS, *extra_algorithms):
        blocks.append(f"\n## {algorithm}\n")
        families_for_algorithm = algorithms.get(algorithm, {})
        visible_families = [
            family for family in VISIBLE_FAMILIES if family in families_for_algorithm
        ]
        if not visible_families:
            blocks.append("TODO: populate from experiments.")
            continue

        blocks.append("::: {.panel-tabset}")
        for family in visible_families:
            label = FAMILY_LABELS.get(family, family)
            blocks.append(f"\n### {label}\n")
            family_rows = families_for_algorithm[family]
            blocks.append(render_raw_table(family_rows))
            blocks.extend(render_notes_toggles(family_rows, family))
        blocks.append(":::")
    blocks.append(":::")
    return "\n".join(blocks)


def update_results_qmd(sections: dict[str, str]) -> None:
    text = RESULTS_QMD.read_text(encoding="utf-8")
    for marker_name, content in sections.items():
        begin_marker = f"<!-- BEGIN AUTO-GENERATED: {marker_name} -->"
        end_marker = f"<!-- END AUTO-GENERATED: {marker_name} -->"
        pattern = re.compile(
            re.escape(begin_marker) + r".*?" + re.escape(end_marker), re.DOTALL
        )
        if not pattern.search(text):
            print(f"warning: markers for {marker_name!r} not found in {RESULTS_QMD}", file=sys.stderr)
            continue
        replacement = f"{begin_marker}\n{content}\n{end_marker}"
        text = pattern.sub(replacement, text)
    RESULTS_QMD.write_text(text, encoding="utf-8")


def main() -> None:
    rows_by_family = collect_rows_by_family()
    verified = load_verified_gs()

    sections = {
        family: render_table(family, rows, verified)
        for family, rows in rows_by_family.items()
    }
    sections["all"] = render_all_section(rows_by_family)
    update_results_qmd(sections)

    for family, rows in sorted(rows_by_family.items()):
        instances = len({(r.n, r.seed) for r in rows})
        print(f"{family}: {len(rows)} row(s) across {instances} instance(s)")


if __name__ == "__main__":
    main()
