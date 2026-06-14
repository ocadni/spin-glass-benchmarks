# Development Testing

This file summarizes the tests that should be run while developing generators,
solver baselines, and `solvers_v2`.

## Environment

Use the pinned CPU solver environment for solver and full-suite checks:

```bash
conda activate /home/biazzin/conda-envs/sgbench-solvers-core
```

If the environment does not exist, create it from the repo root:

```bash
conda env create --prefix /home/biazzin/conda-envs/sgbench-solvers-core -f solvers_v2/envs/core-cpu.yml
```

The exact resolved package set is stored in:

```text
solvers_v2/envs/core-cpu-linux-64.lock
```

## Test Groups

### Generator Tests

These tests validate generators, canonical pairwise instance I/O, and reference
instance snapshots.

```bash
python -m pytest tests/test_generators.py generators/test_*.py
```

Use this when changing `generators/`, instance file formats, or reference
fixture handling.

### Old Solver Baseline Tests

These tests rerun selected old solver functions from `solvers/System specific
solvers/...` and compare their metric histories with JSON baselines under
`tests_data/solver_baselines/`.

```bash
python -m pytest tests/test_solver_baselines.py
```

Use this before and after touching baseline harness code or old solver
compatibility logic. The old solver source files should remain unchanged unless
the baseline files are intentionally regenerated and reviewed.

Current old-code baselines cover:

- SK simulated annealing
- SK population annealing
- SK parallel tempering
- SK greedy
- EA2D simulated annealing
- EA3D simulated annealing
- EA3D population annealing

XORSAT is intentionally deferred until its fixture and parser contract is made
explicit. EA2D population annealing is not baselined because the old population
annealing implementation is hardcoded for 3D indexing.

### `solvers_v2` Parity Tests

These tests run the refactored `solvers_v2` implementations and require them to
match the captured old-code baseline metrics.

```bash
python -m pytest tests/test_solvers_v2_baseline_parity.py
```

Run this whenever changing `solvers_v2/common/` or a family adapter under
`solvers_v2/families/`.

### Full Development Suite

Run the full current development suite from the repo root:

```bash
python -m pytest tests generators/test_*.py
```

This is the command to run before committing solver or generator changes.

## Regenerating Solver Baselines

Regenerate baselines only when intentionally changing the expected old-code
contract. From the repo root, using the pinned solver environment:

```bash
python scripts/generate_solver_baselines.py
```

Then inspect every changed file in:

```text
tests_data/solver_baselines/
```

Baseline JSON files record the old solver path, fixture path, fixture hash,
package versions, seeds, run parameters, schedule length, and expected
`min_energy`/`mean_energy` metrics.

## Reproducibility Rules

- Solver baseline and parity tests run CPU-only.
- Seeds are fixed for Python `random`, NumPy, and PyTorch.
- Compare metric histories, not elapsed time or full final spin populations.
- Do not update baseline JSON files casually; a baseline change means the
  expected solver behavior changed.
- Keep XORSAT out of the parity suite until its input format is normalized.
