# Experimental Framework

A systematic framework for running, tracking, and analyzing spin glass solver experiments.

## Prerequisites

Make sure you have the solver environment installed:

```bash
# For GPU (recommended - 10-100x faster)
conda env create -f environments/solvers-core-gpu.yml
conda activate sgbench-solvers-gpu

# Or for CPU-only
conda env create -f environments/solvers-core-cpu.yml
conda activate sgbench-solvers-core
```

See [environments/README.md](../environments/README.md) for detailed setup instructions.

## Overview

This framework provides:
- **Organized structure**: Hierarchical organization by researcher → experiment → algorithm → runs
- **Machine provenance**: Automatic capture of hardware/software specs for reproducibility
- **Standard + custom results**: Common metrics (energy, runtime) plus algorithm-specific diagnostics
- **Parameter sweeps**: Easy grid or list-based parameter exploration
- **Self-documenting**: Config + results capture everything needed to reproduce

## Directory Structure

```
experiments/
├── src/                             # Framework source code
│   ├── config.py                    # Configuration schemas
│   ├── results.py                   # Result schemas
│   ├── machine_specs.py             # Machine spec collection
│   ├── runner.py                    # Experiment orchestration
│   ├── analysis.py                  # Analysis utilities
│   └── cli.py                       # Command-line interface
├── <researcher>/                    # Your experiments (e.g., "biazzin")
│   ├── <experiment_name>/           # e.g., "sk_ga_ablation_2026"
│   │   ├── experiment_meta.json     # Experiment description
│   │   ├── <algorithm>/             # e.g., "global_annealing"
│   │   │   ├── config.json          # Algorithm configuration
│   │   │   └── runs/                # Results
│   │   │       └── <instance>_seed<seed>/
│   │   │           ├── common.json        # Standard metrics + machine specs
│   │   │           └── diagnostics.json   # Algorithm-specific (optional)
│   │   └── <another_algorithm>/
│   └── <another_experiment>/
└── README.md                        # This file
```

## Quick Start

### 1. Create an Experiment

Create directory structure:
```bash
mkdir -p experiments/yourname/my_experiment/algorithm_name
```

Create `experiments/yourname/my_experiment/experiment_meta.json`:
```json
{
  "schema_version": 1,
  "researcher": "yourname",
  "experiment_name": "my_experiment",
  "description": "Description of what you're testing",
  "created_at": "2026-06-14T16:00:00Z",
  "family": "sk",
  "instances": [
    "instances/sk/N50/sk_couplings_N50_J0_seed1051730.txt"
  ],
  "algorithms": ["simulated_annealing"],
  "tags": ["pilot", "sk"],
  "notes": "Additional notes"
}
```

Create `experiments/yourname/my_experiment/simulated_annealing/config.json`:
```json
{
  "schema_version": 1,
  "algorithm": "simulated_annealing",
  "seeds": [1729, 4242],
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

### 2. Run the Experiment

```bash
python -m experiments.src.cli experiments/yourname/my_experiment/experiment_meta.json
```

### 3. Analyze Results

```python
from pathlib import Path
from experiments.src.analysis import load_experiment

# Load experiment
exp = load_experiment(Path("experiments/yourname/my_experiment"))

# Get results for an algorithm
results = exp.get_algorithm_results("simulated_annealing")

for result in results:
    print(f"Instance: {result.instance_path}")
    print(f"Seed: {result.seed}")
    print(f"Final energy: {result.metrics['final_min_energy']:.4f}")
    print(f"Runtime: {result.runtime_seconds:.2f}s")
    print(f"Machine: {result.machine_specs.hostname}")
    print()
```

## Configuration Reference

### ExperimentConfig (experiment_meta.json)

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Config schema version (always 1) |
| `researcher` | string | Your name or group identifier |
| `experiment_name` | string | Unique experiment identifier |
| `description` | string | What you're testing |
| `created_at` | string | ISO timestamp |
| `family` | string | One of: `sk`, `ea2d`, `ea3d`, `rrg` |
| `instances` | list[string] | Instance paths relative to repo root |
| `algorithms` | list[string] | Algorithms to run |
| `tags` | list[string] | Optional tags for organization |
| `notes` | string | Optional additional notes |

### AlgorithmConfig (algorithm/config.json)

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Config schema version (always 1) |
| `algorithm` | string | Algorithm name |
| `seeds` | list[int] | Random seeds to run |
| `output_diagnostics` | bool | Whether to save diagnostics |
| `solver_parameters` | dict | Base algorithm parameters |
| `parameter_sweep` | object\|null | Optional parameter sweep config |

### Parameter Sweeps

#### Grid Sweep
Explores all combinations (Cartesian product):

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

This generates 6 runs: (50,20), (50,50), (100,20), (100,50), (200,20), (200,50)

#### List Sweep
Parallel iteration (all lists same length):

```json
{
  "parameter_sweep": {
    "sweep_type": "list",
    "parameters": {
      "pop_size": [50, 100, 200],
      "num_temps": [10, 20, 40]
    }
  }
}
```

This generates 3 runs: (50,10), (100,20), (200,40)

## Output Schemas

### common.json
Standard results collected from every run:
- `run_id`: UUID
- `timestamp`: When run completed
- `instance_path`, `instance_sha256`, `num_spins`, `family`: Instance info
- `algorithm`, `seed`: Execution info
- `runtime_seconds`: Wall-clock time
- `metrics`: Dictionary with `min_energy`, `mean_energy`, `final_min_energy`, etc.
- `machine_specs`: Detailed hardware/software specs
- `solver_parameters`: Exact parameters used

### diagnostics.json (optional)
Algorithm-specific diagnostic data. Structure varies by algorithm:

**global_annealing**:
- `made_training`: MADE model training info
  - `initial_training`: Initial training metrics
  - `retraining_losses`: Loss at each temperature
  - `temperature_schedule`: Full temperature schedule

**parallel_tempering**:
- `swap_acceptance_matrix`: Replica exchange statistics
- `replica_temperatures`: Temperature ladder

## Solver Parameters by Algorithm

### simulated_annealing
```json
{
  "pop_size": 100,
  "MCsteps": 100,
  "Tstart": 1.92,
  "Tend": 0.1,
  "num_temps": 50,
  "schedule": "linearT",
  "high_temp_thermalization_steps": 1000
}
```

### population_annealing
```json
{
  "pop_size": 100,
  "MCsteps": 100,
  "Tstart": 1.92,
  "Tend": 0.1,
  "num_temps": 50,
  "schedule": "linearT",
  "high_temp_thermalization_steps": 1000
}
```

### parallel_tempering
```json
{
  "MCsteps": 1000,
  "swap_interval": 10,
  "Tstart": 2.0,
  "Tend": 0.1,
  "num_temps": 20,
  "schedule": "linearT",
  "high_temp_thermalization_steps": 1000
}
```

### greedy
```json
{
  "pop_size": 1000
}
```

### global_annealing
```json
{
  "pop_size": 100,
  "MLMCsteps": 10,
  "swap_step": 2,
  "Tstart": 1.92,
  "Tend": 0.1,
  "num_temps": 20,
  "schedule": "linearT",
  "high_temp_thermalization_steps": 100,
  "num_epochs_start": 40,
  "num_epochs_retrain": 1,
  "batch_size": 256
}
```

## Example Workflows

### Comparing Algorithms

```python
from pathlib import Path
import matplotlib.pyplot as plt
from experiments.src.analysis import load_experiment

exp = load_experiment(Path("experiments/biazzin/sk_ga_initial"))

# Compare two algorithms on same instances
ga_results = exp.get_algorithm_results("global_annealing")
sa_results = exp.get_algorithm_results("simulated_annealing")

ga_energies = [r.metrics['final_min_energy'] for r in ga_results]
sa_energies = [r.metrics['final_min_energy'] for r in sa_results]

plt.scatter(sa_energies, ga_energies)
plt.xlabel("SA Final Energy")
plt.ylabel("GA Final Energy")
plt.title("Algorithm Comparison")
plt.plot([min(sa_energies), max(sa_energies)], 
         [min(sa_energies), max(sa_energies)], 'k--')
plt.show()
```

### Analyzing Parameter Sweep

```python
from pathlib import Path
from experiments.src.analysis import load_experiment

exp = load_experiment(Path("experiments/yourname/sweep_experiment"))
results = exp.get_algorithm_results("simulated_annealing")

# Group by parameter value
by_popsize = {}
for result in results:
    pop_size = result.solver_parameters['pop_size']
    if pop_size not in by_popsize:
        by_popsize[pop_size] = []
    by_popsize[pop_size].append(result.metrics['final_min_energy'])

for pop_size, energies in sorted(by_popsize.items()):
    mean_energy = sum(energies) / len(energies)
    print(f"pop_size={pop_size}: mean_energy={mean_energy:.4f}")
```

### Iterating Over All Runs

```python
from pathlib import Path
from experiments.src.analysis import load_experiment

exp = load_experiment(Path("experiments/biazzin/sk_ga_initial"))

for common, diagnostics in exp.iter_runs("global_annealing"):
    print(f"Run {common.run_id[:8]}:")
    print(f"  Energy: {common.metrics['final_min_energy']:.4f}")
    print(f"  Runtime: {common.runtime_seconds:.2f}s")
    
    if diagnostics:
        training_loss = diagnostics.diagnostics['made_training']['initial_training']['final_loss']
        print(f"  MADE training loss: {training_loss:.4f}")
```

## Programmatic Experiment Creation

```python
from pathlib import Path
from experiments.src.config import ExperimentConfig, AlgorithmConfig

# Create experiment config
exp_config = ExperimentConfig(
    researcher="yourname",
    experiment_name="auto_experiment",
    description="Programmatically created",
    family="sk",
    instances=["instances/sk/N50/sk_couplings_N50_J0_seed1051730.txt"],
    algorithms=["simulated_annealing"],
    created_at=ExperimentConfig.create_timestamp(),
    tags=["automated"],
)

exp_dir = Path("experiments/yourname/auto_experiment")
exp_config.to_file(exp_dir / "experiment_meta.json")

# Create algorithm config
algo_config = AlgorithmConfig(
    algorithm="simulated_annealing",
    seeds=[1729],
    solver_parameters={
        "pop_size": 100,
        "MCsteps": 100,
        "Tstart": 1.92,
        "Tend": 0.1,
        "num_temps": 50,
        "schedule": "linearT",
        "high_temp_thermalization_steps": 1000,
    },
    output_diagnostics=False,
)

algo_dir = exp_dir / "simulated_annealing"
algo_config.to_file(algo_dir / "config.json")
```

## Machine Specs

Each run automatically captures:
- **Hostname**: Machine identifier
- **CPU**: Model, cores (physical/total), frequency
- **Memory**: Total GB
- **GPU**: Name, compute capability, memory (if available)
- **OS**: System, release, version
- **Software**: Python, PyTorch, NumPy versions, CUDA version

This ensures full reproducibility tracking.

## Dependencies

Required:
- `psutil`: Machine specs collection

Optional:
- `py-cpuinfo`: Better CPU info (falls back to platform.processor() if not available)

Install:
```bash
pip install psutil py-cpuinfo
```

## Tips

1. **Start small**: Test with 1-2 instances and small parameter ranges before scaling up
2. **Use descriptive names**: Experiment names should indicate what you're testing
3. **Tag experiments**: Use tags to group related experiments
4. **Document in notes**: Capture context that isn't in parameters
5. **Check baselines**: Verify your configs match baseline tests for reproducibility validation

## Relationship to Baseline Tests

- **Baseline tests** (`tests/test_baseline_parity.py`): Numerical reproducibility validation
- **Experiments** (this framework): Research parameter exploration and comparison

The experiment framework uses the same solver code and family adapters as baseline tests, but adds:
- Flexible parameter configuration
- Machine provenance
- Batch execution
- Analysis utilities
- Hierarchical organization
