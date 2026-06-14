# Solver Testing Guide

This document describes the testing infrastructure for `solvers_v2` - the refactored solver codebase.

## Quick Start

### Environment Setup

Use the pinned CPU solver environment:

```bash
conda activate sgbench-solvers-core
```

If it doesn't exist, create it from the repo root:

```bash
conda env create -f solvers_v2/envs/core-cpu.yml
```

The exact package versions are locked in `solvers_v2/envs/core-cpu-linux-64.lock`.

### Run All Tests

From the repository root:

```bash
python -m pytest tests/ generators/test_*.py -v
```

---

## Test Organization

### Current Test Files

```
tests/
├── test_baseline_parity.py    # Main solver validation (13 tests)
├── test_generators.py          # Generator validation
└── README.md                   # This file
```

### Test Coverage

**13 baseline parity tests** validate all solvers_v2 algorithms:

| Family | Algorithms Tested |
|--------|-------------------|
| **SK** | Simulated Annealing, Population Annealing, Parallel Tempering, Greedy, Global Annealing |
| **EA2D** | Simulated Annealing, Population Annealing |
| **EA3D** | Simulated Annealing, Population Annealing |
| **RRG** | Simulated Annealing, Population Annealing, Parallel Tempering, Greedy |

---

## Baseline Parity Tests

### Purpose

Baseline tests ensure **numerical reproducibility** - the same inputs always produce the same outputs.

### How They Work

1. Load a baseline JSON file from `tests_data/solver_baselines/`
2. Run the solver with the baseline's parameters and seed
3. Compare output metrics (min_energy, mean_energy) to baseline values
4. Fail if any value differs by more than 1e-7

### Run Baseline Tests

```bash
# All baseline tests
python -m pytest tests/test_baseline_parity.py -v

# Single test
python -m pytest tests/test_baseline_parity.py::test_solvers_v2_matches_old_solver_baselines[sk_global_annealing] -v

# Specific family
python -m pytest tests/test_baseline_parity.py -k "sk_" -v
```

### When to Run

- **Before committing** changes to `solvers_v2/`
- After modifying algorithms, updates, schedules, or observables
- After changing family adapters
- When adding new algorithms (create baseline first)

---

## Regenerating Baselines

### ⚠️ Warning

Only regenerate baselines when **intentionally changing solver behavior**. A baseline change means the algorithm now produces different results.

### Generate All Baselines

From the repo root with `sgbench-solvers-core` environment active:

```bash
PYTHONPATH=. python scripts/generate_all_baselines.py
```

This regenerates all 13 baseline JSON files in `tests_data/solver_baselines/`.

### After Regeneration

1. **Review every changed file** using `git diff tests_data/solver_baselines/`
2. Understand **why** values changed
3. Verify changes are intentional (not bugs)
4. Document reason in commit message

### Baseline File Format

Each baseline JSON contains:

```json
{
  "algorithm": "simulated_annealing",
  "baseline_id": "sk_simulated_annealing",
  "family": "sk",
  "fixture": "tests_data/instances/sk/N50/...",
  "fixture_sha256": "7f9f3ee...",
  "parameters": { "pop_size": 8, "MCsteps": 2, ... },
  "seed_values": { "torch": 1729, "numpy": 1729, "python_random": 1729 },
  "schedule_length": 4,
  "metrics": {
    "min_energy": [0.063, 0.098, ...],
    "mean_energy": [0.125, 0.200, ...],
    "best_min_energy": 0.063,
    "final_min_energy": 0.140,
    "final_mean_energy": 0.181
  },
  "environment": { "python": "3.12.2", "numpy": "2.2.3", "torch": "2.7.0" },
  "solver_file": "solvers_v2/families/sk/__init__.py"
}
```

---

## Adding New Algorithms

When you add a new algorithm to `solvers_v2/common/`:

1. **Implement the algorithm** in its own package (e.g., `common/my_algorithm/`)
2. **Add family adapter** in `solvers_v2/families/<family>/__init__.py`
3. **Generate baseline**:
   ```python
   # Add to scripts/generate_all_baselines.py
   generate_baseline(
       family="sk",
       algorithm="my_algorithm",
       fixture=sk_fixture,
       parameters={...},
       seed=1729,
       baseline_id="sk_my_algorithm",
   )
   ```
4. **Run generation script** to create baseline JSON
5. **Verify test passes**: `pytest tests/test_baseline_parity.py -k my_algorithm`

The test framework automatically discovers new baselines from `tests_data/solver_baselines/*.json`.

---

## Reproducibility Rules

### Requirements

- **CPU-only execution** (no GPU for baselines)
- **Fixed seeds** for Python `random`, NumPy, PyTorch
- **Deterministic algorithms** (no true randomness)
- **Pinned environment** (exact package versions)

### What We Test

✅ **Compare**: Metric histories (min_energy, mean_energy over temperature schedule)  
❌ **Don't compare**: Elapsed time, final spin configurations, intermediate states

### Tolerance

- Numerical tolerance: `1e-7` (absolute)
- Schedule length must match exactly
- All metric arrays must have same length

---

## Generator Tests

Generator tests validate instance file creation and I/O:

```bash
python -m pytest tests/test_generators.py generators/test_*.py -v
```

These tests check:
- Canonical pairwise instance format
- Adapter behavior (SK, EA, RRG)
- Deterministic instance generation
- Reference instance snapshots

Run when modifying `generators/` or instance file formats.

---

## Troubleshooting

### Test Fails with "no module named solvers_v2"

Run from repository root with `PYTHONPATH=.`:
```bash
PYTHONPATH=. python -m pytest tests/test_baseline_parity.py
```

### Baseline Mismatch

If a baseline test fails:

1. Check if you modified solver code (expected)
2. Verify seed is set correctly (1729 for all baselines)
3. Check environment matches (`python`, `numpy`, `torch` versions)
4. Run regeneration script if change is intentional
5. Review `git diff` on baseline JSON to understand change

### Import Errors in Tests

Ensure `sgbench-solvers-core` environment is active:
```bash
conda activate sgbench-solvers-core
which python  # Should point to conda env
```

---

## Future Development

### Adding New Families

1. Create family adapter: `solvers_v2/families/new_family/__init__.py`
2. Implement `run_baseline_case()` or `run_pairwise_case()`
3. Add to test dispatcher in `test_baseline_parity.py::_runner_for_family()`
4. Generate baselines for each algorithm
5. Add family to this README

### Performance Benchmarks

Current tests validate **correctness**, not **performance**. For performance benchmarking:
- Use larger instances (N > 1000)
- Measure wall-clock time
- Compare GPU vs CPU
- Profile with `cProfile` or `torch.profiler`

Consider creating separate `tests/performance/` directory for benchmark suites.

---

## References

- Solver implementation: `solvers_v2/`
- Baseline data: `tests_data/solver_baselines/`
- Generation scripts: `scripts/generate_all_baselines.py`
- Environment spec: `solvers_v2/envs/core-cpu.yml`
