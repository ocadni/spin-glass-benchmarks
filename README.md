# Spin Glass Benchmarks

**Curated benchmark instances + reference solvers for spin glass optimization**

📖 **[Full Documentation](https://ocadni.github.io/spin-glass-benchmarks/)** | 
📊 **[View Results](https://ocadni.github.io/spin-glass-benchmarks/results.html)**

---

## What's Inside

- **1,200 SK benchmark instances** (12 sizes × 100 seeds each), full sets on [Hugging Face](https://huggingface.co/datasets/Laplaxe/spin-glass-benchmarks), a 5-per-size sample committed to the repo → [Details](#available-instances)
- **210 EA3D (3D Edwards-Anderson) benchmark instances** (N=1000 and N=2744), taken from [Del Bono, Ricci-Tersenghi & Zamponi, PNAS 2026](https://doi.org/10.1073/pnas.2534768123), full sets on [Hugging Face](https://huggingface.co/datasets/Laplaxe/spin-glass-benchmarks), a 5-per-size sample committed to the repo → [Details](#available-instances)
- **Group submissions**: solver code + benchmark results per group → [Guide](experiments/QUICKSTART.md)
---

## Available Instances

| Family | Description | Sizes | Seeds per size | Location |
|--------|-------------|-------|-----------------|----------|
| **SK** | Sherrington-Kirkpatrick (fully connected) | 50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 2000 | 100 | `instances/sk/` |
| **EA3D** | Edwards-Anderson, 3D cubic lattice, periodic boundaries | 1000, 2744 | 200, 12 | `instances/ea3d/` |

**SK**: 1,200 instances total. Only 5 per size (60 files) are committed to
the repo, as a quick-access sample — the full set lives on
[Hugging Face](https://huggingface.co/datasets/Laplaxe/spin-glass-benchmarks)
and is fetched with the download script below.

**EA3D**: 210 instances total (N=1000: 200 instances, all with exactly known
ground-state energies; N=2744: 10 instances), taken from Del Bono, Luca
Maria, Federico Ricci-Tersenghi, and Francesco Zamponi. "Demonstrating real
advantage of machine learning–enhanced Monte Carlo for combinatorial
optimization." *Proceedings of the National Academy of Sciences* 123.19
(2026): e2534768123. As with SK, only 5 per size are committed to the repo
(N=1000, N=2744); the rest are distributed via the same Hugging Face dataset
(see Downloading Instances below). Only N=1000 and N=2744 are populated (additional N values are planned).

### Instance File Format

Instances are plain-text files. The first line is a `#`-prefixed metadata
header (for example, `model`, `N`, `seed`, `distribution`, `num_fields`, and
`num_couplings`). It is followed by `num_fields` rows of the form `i h_i`,
giving the external field on each zero-based spin index, and then by
`num_couplings` rows of the form `i j J_ij`, giving the coupling between spins
`i` and `j`. The SK instances have zero external fields and list every pair of
spins.

See the [instance file-format specification](https://ocadni.github.io/spin-glass-benchmarks/file_format.html)
for the complete schema, and the [Problems page](https://ocadni.github.io/spin-glass-benchmarks/problem_definition.html)
for the mathematical definitions of the benchmark problem families.

### Downloading Instances

The 5-per-size sample committed to the repo is enough to try things out, but
for the full SK or EA3D sets (or any subset), use
[`instances/download_instances.py`](instances/download_instances.py):

```bash
pip install huggingface_hub

# Everything available (SK + EA3D)
python instances/download_instances.py --type all

# One family
python instances/download_instances.py --type sk
python instances/download_instances.py --type ea3d

# One size (all seeds)
python instances/download_instances.py --type sk --N 1000
python instances/download_instances.py --type ea3d --N 2744

# One specific instance
python instances/download_instances.py --type sk --N 1000 --seed 2001732

# List what's available without downloading anything
python instances/download_instances.py --info
```

Downloaded files land directly under `instances/<family>/N<N>/`, alongside
the committed sample. `--type ea2d` is recognized by the CLI but has no
instances yet, on the Hub or otherwise.

Maintainers adding new instance files to `instances/sk/` or `instances/ea3d/`
should push the full set to the Hub with
[`instances/upload_instances.py`](instances/upload_instances.py) (same
dependency, requires write access to the dataset repo):

```bash
python instances/upload_instances.py --type ea3d
```

---

## Benchmark Annealing Solvers

Reference annealing solvers are included in the Sapienza submission under
[`experiments/sapienza/code/annealing_solvers/`](experiments/sapienza/code/annealing_solvers/).
The easiest entry point is the Modern command-line wrapper:
[`experiments/sapienza/code/annealing_solvers/Modern/optimization/solver.py`](experiments/sapienza/code/annealing_solvers/Modern/optimization/solver.py).
It supports simulated annealing (`sa`), population annealing (`pa`), and
ML-enhanced global annealing (`ga`).

Set up one of the solver environments first:

```bash
# CPU
conda env create -f environments/solvers-core-cpu.yml
conda activate sgbench-solvers-core

# NVIDIA GPU
conda env create -f environments/solvers-core-gpu.yml
conda activate sgbench-solvers-gpu

# Apple Silicon / Metal
conda env create -f environments/solvers-core-mac-gpu.yml
conda activate sgbench-solvers-mac-gpu
```

Run a solver on any downloaded or committed instance file:

```bash
# Simulated annealing
python experiments/sapienza/code/annealing_solvers/Modern/optimization/solver.py \
  instances/ea3d/N1000/ea3d_couplings_N1000_J0_seed418527.txt sa \
  --population-size 256 --num-steps-mc 10 --num-temps 100

# Population annealing
python experiments/sapienza/code/annealing_solvers/Modern/optimization/solver.py \
  instances/ea3d/N1000/ea3d_couplings_N1000_J0_seed418527.txt pa \
  --population-size 256 --num-steps-mc 10 --num-temps 100 \
  --reweight-mode systematic

# ML-enhanced global annealing
python experiments/sapienza/code/annealing_solvers/Modern/optimization/solver.py \
  instances/ea3d/N1000/ea3d_couplings_N1000_J0_seed418527.txt ga \
  --population-size 256 --num-steps-mc 10 --num-temps 100 \
  --swap-step 1 --batch-size 256
```

Common options include `--t-start`, `--t-end`, `--schedule`
(`linearT`, `linearBeta`, or `logT`), `--thermalization-steps`, `--device`
(`auto`, `cpu`, `cuda`, or `mps`), and `--seed`. Add `--json` for
machine-readable output.

By default the wrapper prints a short run summary:

```text
annealer: sa
instance: instances/ea3d/N1000/ea3d_couplings_N1000_J0_seed418527.txt
device: cpu
spins: 1000
external fields: zero
temperatures: 100 (3.0 -> 0.2)
final minimum energy/spin: -1.234567
best minimum energy/spin: -1.234567
total elapsed time: 12.345 s
```

For `ga`, the output also separates `training time` and `annealing time`.
With `--json`, the same information is emitted as JSON fields such as
`annealer`, `instance`, `device`, `num_spins`, `has_external_fields`,
`num_temperatures`, `final_min_energy_per_spin`,
`best_min_energy_per_spin`, and `elapsed_seconds`.

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
