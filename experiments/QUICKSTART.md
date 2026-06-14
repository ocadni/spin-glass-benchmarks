# Experiments Quick Reference

## Structure

```
experiments/
├── src/                  # Framework code (don't modify)
├── <your_name>/          # Your experiments go here
│   └── <experiment>/
│       ├── experiment_meta.json
│       └── <algorithm>/
│           ├── config.json
│           └── runs/     # Results appear here
└── README.md            # Full documentation
```

## Run an Experiment

**Option 1: Using the template script (recommended)**
```bash
# Copy template to your experiment directory
cp experiments/run_experiment_template.sh experiments/yourname/my_experiment/run_experiment.sh
chmod +x experiments/yourname/my_experiment/run_experiment.sh

# Run it
./experiments/yourname/my_experiment/run_experiment.sh
```

**Option 2: Direct command**
```bash
# For GPU (set CUBLAS config for deterministic operations)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m experiments.src.cli experiments/yourname/experiment_name/experiment_meta.json

# For CPU
python -m experiments.src.cli experiments/yourname/experiment_name/experiment_meta.json
```

## Create New Experiment

```bash
# 1. Create structure
mkdir -p experiments/yourname/my_experiment/algorithm_name

# 2. Create experiment_meta.json (see template below)
# 3. Create algorithm_name/config.json (see template below)
# 4. Run it!
```

## Minimal experiment_meta.json

```json
{
  "schema_version": 1,
  "researcher": "yourname",
  "experiment_name": "my_experiment",
  "description": "What I'm testing",
  "created_at": "2026-06-14T16:00:00Z",
  "family": "sk",
  "instances": ["instances/sk/N50/sk_couplings_N50_J0_seed1051730.txt"],
  "algorithms": ["simulated_annealing"],
  "tags": [],
  "notes": ""
}
```

## Minimal algorithm config.json

```json
{
  "schema_version": 1,
  "algorithm": "simulated_annealing",
  "seeds": [1729],
  "output_diagnostics": false,
  "solver_parameters": {
    "pop_size": 100,
    "MCsteps": 100,
    "Tstart": 1.92,
    "Tend": 0.1,
    "num_temps": 50,
    "schedule": "linearT",
    "high_temp_thermalization_steps": 1000
  },
  "parameter_sweep": null
}
```

## Analyze Results

```python
from pathlib import Path
from experiments.src.analysis import load_experiment

exp = load_experiment(Path("experiments/yourname/my_experiment"))
results = exp.get_algorithm_results("simulated_annealing")

for r in results:
    print(f"{r.instance_path}: {r.metrics['final_min_energy']:.4f}")
```

## Parameter Sweep (Grid)

Add to algorithm config.json:

```json
{
  "parameter_sweep": {
    "sweep_type": "grid",
    "parameters": {
      "pop_size": [50, 100, 200],
      "num_temps": [20, 50]
    }
  }
}
```

This runs 6 combinations: (50,20), (50,50), (100,20), (100,50), (200,20), (200,50)

## Families

- `"sk"`: Sherrington-Kirkpatrick
- `"ea2d"`: 2D Edwards-Anderson
- `"ea3d"`: 3D Edwards-Anderson  
- `"rrg"`: Random Regular Graph

## Algorithms

- `simulated_annealing`
- `population_annealing`
- `parallel_tempering`
- `greedy`
- `global_annealing`

See [README.md](README.md) for full parameter specifications.

## Full Documentation

See [experiments/README.md](README.md) for:
- Complete configuration reference
- All algorithm parameters
- Analysis examples
- Advanced features
