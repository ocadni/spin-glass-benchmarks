# Experimental Framework Implementation Summary

## Overview

A complete experimental framework has been implemented for the spin glass solver codebase. This framework enables systematic execution, tracking, and analysis of solver experiments with full reproducibility tracking.

## What Was Implemented

### 1. Core Framework (experiments/src/)

**Data Structures:**
- `config.py`: Configuration schemas (ExperimentConfig, AlgorithmConfig, ParameterSweep)
- `results.py`: Result schemas (CommonResult, DiagnosticsResult, MachineSpecs)
- `machine_specs.py`: Automatic hardware/software specification collection
- `runner.py`: Experiment orchestration and execution
- `analysis.py`: Result loading and analysis utilities
- `cli.py`: Command-line interface

**Features:**
- Grid and list parameter sweeps
- Automatic machine provenance tracking
- JSON-based configuration and results
- Hierarchical organization: researcher → experiment → algorithm → runs

Usage:
```bash
python -m experiments.src.cli path/to/experiment_meta.json
```

### 3. Solver Integration

**Modified:** `solvers_v2/src/result.py`
- Added optional `diagnostics: dict | None = None` field to SolverResult
- Fully backward compatible with existing code

**Integration points:**
- Uses existing family adapters: `run_baseline_case()`, `run_pairwise_case()`
- Leverages existing SolverResult and Observables classes
- No changes required to baseline tests

### 4. Example Experiment

**Created:** `experiments/biazzin/sk_ga_initial/`
- Demonstrates framework structure
- Includes configs for global_annealing and simulated_annealing
- 2 SK instances, 2 seeds each = 8 total runs

Directory structure:
```
experiments/biazzin/sk_ga_initial/
├── experiment_meta.json
├── global_annealing/
│   └── config.json
└── simulated_annealing/
    └── config.json
```

### 5. Documentation

**Created:** `experiments/README.md`
- Comprehensive usage guide
- Configuration reference
- Parameter specifications for all algorithms
- Example workflows and analysis code
- Quick start guide

**Created:** `experiments/verify_framework.py`
- Verification script to check framework installation
- No dependencies required
- Validates structure and configuration

## Directory Structure

```
experiments/
├── .gitignore                       # Ignore run results
├── README.md                        # Documentation
├── verify_framework.py              # Verification script
├── src/                             # Framework source code
│   ├── __init__.py
│   ├── config.py
│   ├── results.py
│   ├── machine_specs.py
│   ├── runner.py
│   ├── analysis.py
│   └── cli.py
└── biazzin/                         # Example experiment
    └── sk_ga_initial/
        ├── experiment_meta.json
        ├── global_annealing/
        │   ├── config.json
        │   └── runs/                # Results go here
        └── simulated_annealing/
            ├── config.json
            └── runs/                # Results go here
```

## Results Schema

### common.json (every run)
- Run metadata (ID, timestamp)
- Instance info (path, hash, size, family)
- Execution info (algorithm, seed, runtime)
- Standard metrics (energy histories, final values)
- **Machine specs** (CPU, GPU, memory, OS, software versions)
- Solver parameters used

### diagnostics.json (optional)
- Algorithm-specific diagnostic data
- Flexible structure varies by algorithm
- Examples: MADE training losses, swap acceptance rates

## Key Features

### 1. Machine Provenance
Automatically captures:
- Hostname
- CPU model, cores, frequency
- Memory
- GPU (name, compute capability, memory)
- OS details
- Python, PyTorch, NumPy, CUDA versions

### 2. Parameter Sweeps

**Grid sweep** (Cartesian product):
```json
{
  "sweep_type": "grid",
  "parameters": {
    "pop_size": [50, 100, 200],
    "num_temps": [20, 50]
  }
}
```
Generates: 6 runs (all combinations)

**List sweep** (parallel):
```json
{
  "sweep_type": "list",
  "parameters": {
    "pop_size": [50, 100, 200],
    "num_temps": [10, 20, 40]
  }
}
```
Generates: 3 runs (paired values)

### 3. Analysis Utilities

Load experiment:
```python
from experiments.src.analysis import load_experiment

exp = load_experiment(Path("experiments/biazzin/sk_ga_initial"))
results = exp.get_algorithm_results("global_annealing")

for result in results:
    print(f"Energy: {result.metrics['final_min_energy']:.4f}")
    print(f"Runtime: {result.runtime_seconds:.2f}s")
    print(f"Machine: {result.machine_specs.hostname}")
```

## Dependencies

**Required:**
- `psutil`: Machine specs collection

**Optional:**
- `py-cpuinfo`: Better CPU information (falls back to platform.processor())

Install:
```bash
pip install psutil py-cpuinfo
```

## Usage Example

### 1. Create experiment structure
```bash
mkdir -p experiments/yourname/my_experiment/simulated_annealing
```

### 2. Create configs
See `experiments/README.md` for full examples.

### 3. Run experiment
```bash
python -m experiments.src.cli experiments/yourname/my_experiment/experiment_meta.json
```

### 4. Analyze results
```python
from experiments.src.analysis import load_experiment
exp = load_experiment(Path("experiments/yourname/my_experiment"))
results = exp.get_algorithm_results("simulated_annealing")
```

## Verification

Run verification script:
```bash
python experiments/verify_framework.py
```

This checks:
- All framework files present
- Example experiment properly configured
- SolverResult modification applied

## Integration with Existing Code

### Clean Separation
- **Baseline tests** (`tests/`): Numerical reproducibility validation
- **Experiments** (`experiments/`): Research parameter exploration

### Minimal Changes
1. Added optional `diagnostics` field to `SolverResult` (backward compatible)
2. All new code in `experiments/` directory
3. Uses existing solver implementations via family adapters
4. No changes to baseline tests or core solver logic

### Future Extensions
The framework is designed for easy extension:
- Add diagnostics to specific algorithms (modify algorithm code to populate `diagnostics` dict)
- Parallel execution (modify runner to use multiprocessing)
- Resume capability (check for existing results before running)
- Visualization tools (build on analysis utilities)
- Database storage (replace JSON with SQLite)

## Files Created

**Framework core:**
- experiments/src/__init__.py
- experiments/src/config.py
- experiments/src/results.py
- experiments/src/machine_specs.py
- experiments/src/runner.py
- experiments/src/analysis.py
- experiments/src/cli.py

**Documentation:**
- experiments/README.md
- experiments/verify_framework.py
- EXPERIMENTS_IMPLEMENTATION.md (this file)

**Example:**
- experiments/biazzin/sk_ga_initial/experiment_meta.json
- experiments/biazzin/sk_ga_initial/global_annealing/config.json
- experiments/biazzin/sk_ga_initial/simulated_annealing/config.json

**Other:**
- experiments/.gitignore

**Modified:**
- solvers_v2/src/result.py (added optional diagnostics field)

## Next Steps

1. **Install dependencies:**
   ```bash
   pip install psutil py-cpuinfo
   ```

2. **Run example experiment:**
   ```bash
   python -m experiments.cli.run_experiment experiments/biazzin/sk_ga_initial/experiment_meta.json
   ```

3. **Verify results:**
   - Check `experiments/biazzin/sk_ga_initial/*/runs/` directories
   - Each run should have `common.json`
   - global_annealing runs should have `diagnostics.json` (once diagnostics are implemented)

4. **Optional: Add diagnostics to algorithms:**
   - Modify `solvers_v2/common/global_annealing/__init__.py` to track MADE training
   - Modify `solvers_v2/common/parallel_tempering/__init__.py` to track swap stats
   - Return diagnostics in SolverResult

5. **Create your own experiments:**
   - Follow examples in `experiments/README.md`
   - Use parameter sweeps to explore solver behavior
   - Analyze results with provided utilities

## Success Criteria

✓ Framework structure implemented
✓ Configuration schemas defined
✓ Machine specs collection implemented
✓ Experiment runner implemented
✓ Analysis utilities implemented
✓ CLI interface created
✓ Example experiment configured
✓ Comprehensive documentation written
✓ Backward compatibility maintained
✓ Verification script passes

The experimental framework is complete and ready for use!
