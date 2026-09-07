# Experiments Quick Reference

## Structure

```
experiments/
└── <your_group>/
    ├── code/                 # Your solver code
    └── results/
        └── <family>/         # sk, ea2d, ea3d, or rrg
            ├── summary.csv   # Mandatory columns, see below
            └── notes.md      # Optional
```

See [experiments/sapienza/](sapienza/) for a working example.

## Submit results

1. Create `experiments/<your_group>/code/` and put your solver code there
   (any layout or language).
2. Create `experiments/<your_group>/results/<family>/summary.csv` with one
   row per run. Mandatory columns:

   `N, seed, min_energy, average_time, success_probability, TTS, hardware, program_name`

   You may add extra columns (e.g. `average_steps`, as `sapienza` does).
3. Optionally add `experiments/<your_group>/results/<family>/notes.md` for
   context that doesn't fit in the CSV.
4. Regenerate the site tables:

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
