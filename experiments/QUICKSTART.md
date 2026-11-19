# Experiments Quick Reference

## Structure

```
experiments/
└── <your_group>/             # Shown as the Uploader on the site
    ├── README.md             # Description of your algorithms
    ├── code/                 # Your solver code
    └── results/
        └── <family>/         # sk, ea2d, ea3d, or rrg
            ├── summary.csv   # Mandatory columns, see below
            └── notes.md      # Optional
```

See [experiments/sapienza/](sapienza/) for a working example.

## Submit results

1. Create `experiments/<your_group>/code/` and put your solver code there
   (any layout or language). The folder name `<your_group>` is the uploader
   name shown, with a link to the folder, in the results tables.
2. Describe each algorithm (definition, rationale, configuration of the
   reported runs) in `experiments/<your_group>/README.md`, following
   [sapienza/README.md](sapienza/README.md), and add one row per algorithm to
   the table in `docs/algorithms.qmd`.
3. Create `experiments/<your_group>/results/<family>/summary.csv` with one
   row per run. Mandatory columns:

   `N, seed, min_energy, average_time, success_probability, TTS, hardware, program_name`

   You may add extra columns (e.g. `average_steps`, as `sapienza` does).
4. Optionally add `experiments/<your_group>/results/<family>/notes.md` for
   context that doesn't fit in the CSV.
5. Regenerate the site tables:

   ```bash
   python scripts/generate_results_tables.py
   quarto render docs --to html
   ```

## Families

- `sk`: Sherrington-Kirkpatrick
- `ea2d`: 2D Edwards-Anderson
- `ea3d`: 3D Edwards-Anderson
- `rrg`: Random Regular Graph

## Full Documentation

See [experiments/README.md](README.md) for the complete `summary.csv`
column reference and how results reach the site.
