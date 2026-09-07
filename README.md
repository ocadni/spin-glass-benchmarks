# Spin Glass Benchmarks

**Curated benchmark instances + reference solvers for spin glass optimization**

📖 **[Full Documentation](https://ocadni.github.io/spin-glass-benchmarks/)** | 
📊 **[View Results](https://ocadni.github.io/spin-glass-benchmarks/results.html)**

---

## What's Inside

- **1,200 SK benchmark instances** (12 sizes × 100 seeds each), full sets on [Hugging Face](https://huggingface.co/datasets/Laplaxe/spin-glass-benchmarks), a 5-per-size sample committed to the repo → [Details](#available-instances)
- **Group submissions**: solver code + benchmark results per group → [Guide](experiments/QUICKSTART.md)
---

## Available Instances

| Family | Description | Sizes | Seeds per size | Location |
|--------|-------------|-------|-----------------|----------|
| **SK** | Sherrington-Kirkpatrick (fully connected) | 50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 2000 | 100 | `instances/sk/` |

**Total**: 1,200 SK instances. Only 5 per size (60 files) are committed to
the repo, as a quick-access sample — the full set lives on
[Hugging Face](https://huggingface.co/datasets/Laplaxe/spin-glass-benchmarks)
and is fetched with the download script below.

### Downloading Instances

The 5-per-size sample committed to the repo is enough to try things out, but
for the full 1,200-instance SK set (or any subset of it), use
[`instances/download_instances.py`](instances/download_instances.py):

```bash
pip install huggingface_hub

# Everything available (currently: all SK instances)
python instances/download_instances.py --type all

# One family
python instances/download_instances.py --type sk

# One size (all 100 seeds)
python instances/download_instances.py --type sk --N 1000

# One specific instance
python instances/download_instances.py --type sk --N 1000 --seed 2001732

# List what's available without downloading anything
python instances/download_instances.py --info
```

Downloaded files land directly under `instances/<family>/N<N>/`, alongside
the committed sample. `--type ea2d`/`ea3d` are recognized but not yet
available for download (no instances exist yet — see above).

---

## Submitting Results

`experiments/<group_name>/` is where each participating group submits their
solver code (`code/`) and benchmark results (`results/<family>/summary.csv`,
with an optional `notes.md`). Mandatory `summary.csv` columns: `N`, `seed`,
`min_energy`, `average_time`, `success_probability`, `TTS`, `hardware`,
`program_name` — extra columns are allowed. See
[experiments/sapienza/](experiments/sapienza/) for a working example.

Once a `summary.csv` is added or updated, regenerate the site tables:
```bash
python scripts/generate_results_tables.py
quarto render docs --to html
```

[Full guide →](experiments/QUICKSTART.md) | [Submission reference →](experiments/README.md)

---

## Generating Instances

Instances are **pre-generated** and committed to the repository. To create new ones:

```bash
# Setup
conda env create -f environments/generators.yml
conda activate sgbench-generators

# Generate single instance
python generators/generator.py sk 100 --seed 12345 --outdir instances/sk

# Regenerate all systematic instances
bash scripts/generate_random_seed_systematic.sh

# Verify integrity
python -m pytest tests/generators/test_reference_instances.py
```

[Generator documentation →](https://ocadni.github.io/spin-glass-benchmarks/implementation/instances.html#generation-methodology)