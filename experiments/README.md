# Experiments

`experiments/` holds each participating group's submission: their solver code
and the results they want included in the [benchmark results
site](https://ocadni.github.io/spin-glass-benchmarks/results.html).
[`scripts/generate_results_tables.py`](../scripts/generate_results_tables.py)
scans every group's `results/<family>/summary.csv` and regenerates the tables
in [docs/results.qmd](../docs/results.qmd).

## Directory layout

```
experiments/
└── <group_name>/             # e.g. "sapienza"
    ├── code/                 # The group's solver code (any layout/language)
    │   └── ...
    └── results/
        └── <family>/         # sk, ea2d, ea3d, or rrg
            ├── summary.csv   # Mandatory — one row per run, see below
            └── notes.md      # Optional — additional notes for this family
```

`experiments/sapienza/` is a working example: `code/greedy_code/` holds the
solver source and the scripts used to produce results, and
`results/sk/summary.csv` + `results/sk/notes.md` hold the submitted SK
results and notes.

## Submitting results: summary.csv

Each `results/<family>/summary.csv` is a plain CSV with one row per run.
These columns are mandatory:

| Column | Meaning |
|---|---|
| `N` | System size (number of spins) |
| `seed` | Instance seed |
| `min_energy` | Best energy (per spin) found |
| `average_time` | Mean wall-clock runtime, in seconds |
| `success_probability` | Fraction of runs reaching the target energy |
| `TTS` | Time-to-solution |
| `hardware` | Hardware the run was performed on |
| `program_name` | Name of the algorithm/program that produced the row |

You can add further columns beyond these. For example, `sapienza`'s
`summary.csv` includes an optional `average_steps` column (mean spin flips
per replica), which the results tables display when present and as an em
dash otherwise. Use `experiments/sapienza/results/sk/summary.csv` as a
reference for the expected format.

Another optional column is `parameters`: a free-text, comma-separated
`name=value` summary of the run's solver settings, shown as-is in the "All"
section's Parameters column (the Leaderboard ignores it). `sapienza`'s
Global Annealing (GA) and Population Annealing (PA) submissions populate it;
[`scripts/generate_results_tables.py`](../scripts/generate_results_tables.py)
abbreviates their parameter names for display:

| Raw name (in `summary.csv`) | Displayed as | Meaning |
|---|---|---|
| `global_steps_per_temperature` (GA only) | $\theta_g$ | Number of global (population-level) Monte Carlo steps performed at each temperature |
| `MCS_per_global_steps` (GA only) | $\theta_l$ | Number of local (single-spin) Monte Carlo sweeps performed between global steps |
| `MCS_per_temperature` (PA only) | $\theta_l$ | Number of local (single-spin) Monte Carlo sweeps performed at each temperature |
| `number_of_temperatures` | Num. temp. | Number of temperatures in the annealing schedule |
| `schedule` | Schedule | Name of the temperature schedule (e.g. `logT` for logarithmic spacing) |

## Notes

A group can add a `notes.md` next to a family's `summary.csv` (e.g.
`experiments/sapienza/results/sk/notes.md`) for any additional context —
methodology, caveats, anything not captured by the CSV columns. If present
and non-empty, it's shown as a collapsible toggle next to that family's
results in the "All" section of the results site.

## Regenerating the results tables

After adding or updating a `summary.csv`, regenerate the site tables:

```bash
python scripts/generate_results_tables.py
quarto render docs --to html
```
