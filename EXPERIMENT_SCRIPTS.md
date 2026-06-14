# Experiment Run Scripts

## Overview

Each experiment can have a `run_experiment.sh` script that handles:
- Environment activation
- GPU detection and configuration
- Running the experiment
- Displaying results location

## Example Script

See `experiments/biazzin/sk_ga_initial/run_experiment.sh` for a working example.

**Run it:**
```bash
./experiments/biazzin/sk_ga_initial/run_experiment.sh
```

**What it does:**
1. Checks GPU availability via `nvidia-smi`
2. Finds the conda environment
3. Sets CUDA deterministic configuration (`CUBLAS_WORKSPACE_CONFIG=:4096:8`)
4. Runs the experiment
5. Shows results location

## Create Your Own Script

### Option 1: Copy Template

```bash
# Copy template to your experiment directory
cp experiments/run_experiment_template.sh experiments/yourname/my_experiment/run_experiment.sh

# Make executable
chmod +x experiments/yourname/my_experiment/run_experiment.sh

# Run it
./experiments/yourname/my_experiment/run_experiment.sh
```

### Option 2: Minimal Script

For simple cases, create a minimal script:

```bash
#!/bin/bash
set -e

# Set CUDA deterministic config (for GPU)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Run experiment
cd "$REPO_ROOT"
conda run -n sgbench-solvers-gpu python -m experiments.src.cli "$SCRIPT_DIR/experiment_meta.json"
```

## Template Features

The full template (`experiments/run_experiment_template.sh`) includes:

### Configuration
```bash
ENVIRONMENT_NAME="sgbench-solvers-gpu"  # or sgbench-solvers-core
USE_GPU=true  # Set to false for CPU-only
```

### GPU Detection
- Checks `nvidia-smi` availability
- Displays GPU name and memory
- Sets deterministic CUDA configuration

### Error Handling
- Checks if environment exists
- Checks if `experiment_meta.json` exists
- Displays helpful error messages

### Output
- Color-coded status messages
- Experiment configuration summary
- Results location on success
- Clear error messages on failure

## CUDA Deterministic Configuration

For reproducible GPU runs, the script sets:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

This is required when `torch.use_deterministic_algorithms(True)` is enabled in the solvers (which ensures bit-for-bit reproducibility).

**Without this:** You'll get errors like:
```
ERROR: Deterministic behavior was enabled but this operation is not deterministic because it uses CuBLAS
```

**With this:** GPU operations are deterministic (but slightly slower).

## Script Location

Place scripts in your experiment directory:
```
experiments/yourname/my_experiment/
├── experiment_meta.json
├── run_experiment.sh        # ← Your script here
├── algorithm1/
│   └── config.json
└── algorithm2/
    └── config.json
```

## Direct Command Alternative

If you don't want to use a script, run directly:

**GPU:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m experiments.src.cli experiments/yourname/my_experiment/experiment_meta.json
```

**CPU:**
```bash
python -m experiments.src.cli experiments/yourname/my_experiment/experiment_meta.json
```

## Benefits of Using Scripts

1. **Self-documenting** - Script shows how to run the experiment
2. **Environment checks** - Verifies GPU, conda environment
3. **Easy to share** - Others can run with one command
4. **Consistent setup** - Same CUDA config every time
5. **Better errors** - Helpful messages if something is missing

## See Also

- [Experiments Framework](experiments/README.md) - Full documentation
- [Quick Reference](experiments/QUICKSTART.md) - Command cheat sheet
- [Environments Guide](environments/README.md) - Setup instructions
- [Example Experiment](experiments/biazzin/sk_ga_initial/README.md) - Working example
