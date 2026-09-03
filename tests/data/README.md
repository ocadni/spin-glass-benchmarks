# Test Data

Reference data for development-time testing.

## Directory Structure

```
tests/data/
├── instances/              # Small test instances
│   ├── sk/
│   ├── ea2d/
│   ├── ea3d/
│   └── rrg/
└── solver_baselines/       # Numerical reference baselines (13 files)
    ├── sk_*.json           # SK family (5 algorithms)
    ├── ea2d_*.json         # EA2D family (2 algorithms)
    ├── ea3d_*.json         # EA3D family (2 algorithms)
    └── rrg_*.json          # RRG family (4 algorithms)
```

---

## Instances

### Purpose

Small reference instances for **fast testing** (< 5 seconds per test).

These are **snapshots** from the main `instances/` directory, providing stable test data independent of instance generation changes.

### Contents

| Family | Sizes | Count | Example |
|--------|-------|-------|---------|
| **SK** | N=50, 100, 150, 200, 300 | 50 | `sk_couplings_N50_J0_seed1051730.txt` |
| **EA2D** | N=100, 256, 1024, 2304 | 40 | `ea2d_couplings_N100_J0_seed2011730.txt` |
| **EA3D** | N=512, 1000, 1728, 2744 | 40 | `ea3d_couplings_N512_J0_seed3009730.txt` |
| **RRG** | N=50, 100, 150, 200, 300 | 50 | `rrg_couplings_N50_J0_seed4051730.txt` |

### Usage

Tests reference these via relative paths:
```python
ROOT = Path(__file__).resolve().parents[1]
fixture = ROOT / "tests/data/instances/sk/N50/sk_couplings_N50_J0_seed1051730.txt"
```

---

## Solver Baselines

### Purpose

Numerical **reference data** that captures expected solver behavior. Used to detect unintended changes.

### Current Baselines (13 total)

**SK Family (5):**
- `sk_simulated_annealing.json`
- `sk_population_annealing.json`
- `sk_parallel_tempering.json`
- `sk_greedy.json`
- `sk_global_annealing.json` ⭐ ML-enhanced

**EA2D Family (2):**
- `ea2d_simulated_annealing.json`
- `ea2d_population_annealing.json`

**EA3D Family (2):**
- `ea3d_simulated_annealing.json`
- `ea3d_population_annealing.json`

**RRG Family (4):**
- `rrg_simulated_annealing.json`
- `rrg_population_annealing.json`
- `rrg_parallel_tempering.json`
- `rrg_greedy.json`

### Baseline Schema

Each JSON file contains:

```json
{
  "schema_version": 1,
  "baseline_id": "sk_simulated_annealing",
  "algorithm": "simulated_annealing",
  "family": "sk",
  
  "fixture": "tests/data/instances/sk/N50/...",
  "fixture_sha256": "7f9f3eedc6e7f926...",
  
  "parameters": {
    "pop_size": 8,
    "MCsteps": 2,
    "Tstart": 1.92,
    "Tend": 0.1,
    "num_temps": 4,
    "schedule": "linearT",
    "high_temp_thermalization_steps": 1
  },
  
  "seed_values": {
    "torch": 1729,
    "numpy": 1729,
    "python_random": 1729
  },
  
  "schedule_length": 4,
  
  "metrics": {
    "min_energy": [0.063, 0.098, 0.109, 0.140, 0.140],
    "mean_energy": [0.125, 0.200, 0.192, 0.181, 0.181],
    "final_min_energy": 0.140,
    "final_mean_energy": 0.181,
    "best_min_energy": 0.063
  },
  
  "environment": {
    "python": "3.12.2",
    "numpy": "2.2.3",
    "torch": "2.7.0",
    "platform": "Linux-5.14.0-..."
  },
  
  "solver_file": "solvers_v2/families/sk/__init__.py",
  "matrix_mode": "sk_legacy_upper"
}
```

### How Baselines Are Used

1. **Load baseline** from JSON
2. **Run solver** with baseline's fixture, parameters, and seed
3. **Compare metrics** (min_energy, mean_energy arrays)
4. **Fail if mismatch** > 1e-7 tolerance

This ensures solvers produce **identical numerical results** across:
- Code refactors
- Environment changes
- Different machines

### Regenerating Baselines

⚠️ **Only regenerate when intentionally changing solver behavior**

```bash
# From repo root with sgbench-solvers-core environment
PYTHONPATH=. python scripts/generate_all_baselines.py
```

Then:
1. Review `git diff tests/data/solver_baselines/`
2. Verify changes are intentional
3. Document reason in commit message

---

## Version Control

### What to Commit

✅ **Do commit:**
- Baseline JSON files (when intentionally updated)
- New baselines for new algorithms
- Instance files (rarely change)

❌ **Don't commit:**
- Temporary test outputs
- Generated plots
- Large instance files (use main `instances/` directory)

### When Baselines Change

Baseline changes should be **rare** and **intentional**:

**Valid reasons:**
- Algorithm improvement (better energy found)
- Bug fix (correcting wrong behavior)
- New algorithm added
- Numerical precision change

**Invalid reasons:**
- "Tests were failing so I regenerated"
- Random seed changed accidentally
- Environment version mismatch
- Non-deterministic code introduced

---

## Maintenance

### Adding New Test Instances

1. Generate instance in main `instances/` directory
2. Copy small example (N ≤ 200) to `tests/data/instances/<family>/`
3. Use in baseline generation script
4. Commit both instance and baseline

### Removing Old Baselines

When retiring an algorithm:
1. Remove baseline JSON file
2. Update this README
3. Remove generation code from `scripts/generate_all_baselines.py`

Tests automatically discover baselines via `glob("*.json")`, so removal is immediate.

---

## File Formats

### Instance Files

See `generators/pairwise_io.py` for canonical format:

```
# model=sk N=50 meanJ=0 seed=1051730 ...
0 0.123        # field for spin 0
1 -0.456       # field for spin 1
...
0 1 0.789      # coupling between spins 0-1
1 2 -0.234     # coupling between spins 1-2
...
```

### Baseline Files

JSON with deterministic field order (Python 3.7+ dict ordering).

**Critical fields:**
- `metrics`: Must match exactly (arrays + scalars)
- `schedule_length`: Must match exactly
- `fixture_sha256`: Ensures instance hasn't changed
- `seed_values`: Ensures reproducibility

**Informational fields:**
- `environment`: For debugging version mismatches
- `solver_file`: Tracks which implementation created baseline

---

## References

- Test runner: `tests/test_baseline_parity.py`
- Generation script: `scripts/generate_all_baselines.py`
- Instance generators: `generators/`
- Solver implementation: `solvers_v2/`
