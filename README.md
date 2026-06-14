# Spin Glass Benchmarks

A collection of benchmark instances and solvers for various spin glass models. This repository aims to provide a standardized set of tools for researchers and developers working on optimization problems related to spin glasses.

## Models

The Hamiltonian for the models discussed here is generally given by:
H = - sum_{<i,j>} J_{ij} s_i s_j
where `s_i` are Ising spins (s_i in {-1, 1}) and `J_{ij}` are coupling constants.

### Edwards-Anderson (EA) Model

The Edwards-Anderson (EA) model is defined on a d-dimensional lattice (typically d=2 or d=3) with nearest-neighbor interactions. The coupling constants `J_{ij}` are drawn from a random distribution, commonly a bimodal distribution (+-J) or a Gaussian distribution. This model is used to study disordered magnetic systems and features a complex energy landscape.

### Sherrington-Kirkpatrick (SK) Model

The Sherrington-Kirkpatrick (SK) model is a mean-field model where every spin is connected to every other spin. It is defined on a complete graph. The coupling constants `J_{ij}` are typically drawn from a Gaussian distribution with zero mean and a variance that scales with the number of spins `N` (e.g., Var(J_ij) = 1/N). The SK model is exactly solvable in the thermodynamic limit (N -> infinity) but finding the ground state for a finite instance is NP-hard.

### Random Regular Graph (RRG) Model

This model is defined on a random graph where every node (spin) has the same degree `k`. It can be seen as an intermediate between the finite-dimensional EA model and the infinite-dimensional SK model. The couplings `J_{ij}` for existing edges are drawn from a random distribution, similar to the EA model.

## Repository Structure

- `benchmarks/`: Contains benchmark instance files.
- `data/`: For storing generated or experimental data.
- `docs/`: Documentation, including the file format specification.
- `generators/`: Python scripts to generate instances for different models.
- `solvers/`: Implementations of various spin glass solvers (e.g., Simulated Annealing, Population Annealing).
- `scripts/`: Utility scripts for running experiments, analyzing results, etc.
- `.github/`: CI/CD workflows.

## Getting Started

1.  **Generate an instance:**
    ```bash
    python generators/generator.py sk 100 --seed 1 --outdir benchmarks/sk --meanJ 0 --distribution gaussian --field 0
    ```
    This writes an instance such as `benchmarks/sk/N100/sk_couplings_N100_J0_seed1.txt`.

2.  **Run a solver:**
    ```bash
    # (Assuming solver is adapted to read from file)
    python solvers/sa.py benchmarks/sk/N100/sk_couplings_N100_J0_seed1.txt
    ```

## Generator Reproducibility and Tests

The systematic generation script uses deterministic seeds, and the generator
environment pins Python and NumPy because exact random-number streams can depend
on their versions.

Create the generator environment once:

```bash
conda env create -f generators/environment.yml
```

If the environment already exists, update it to the pinned versions:

```bash
conda env update -f generators/environment.yml --prune
```

Then generate the systematic instance set:

```bash
conda activate sgbench-generators
bash scripts/generate_random_seed_systematic.sh
```

For the checked-in deterministic setup, the generator runtime should be:

```bash
python -c 'import sys, numpy as np; print(sys.version.split()[0]); print(np.__version__)'
```

Expected output:

```text
3.12.2
2.2.3
```

To reproduce the exact same instance files, keep the generator code, the seed
formula in `scripts/generate_random_seed_systematic.sh`, and
`generators/environment.yml` fixed. Existing files from older random runs are not
deleted by the script, so remove or archive them separately if you need a clean
deterministic instance directory.

The generator test suite covers the canonical instance representation, file I/O,
adapter behavior, and deterministic reference data:

- `generators/test_reference_instances.py` compares the files already generated
  under `instances/` against the reference snapshot in `tests_data/instances/`.
  It fails on missing files, unexpected files, or any byte-level content change.
- `generators/test_generator_adapters.py` checks that the unified
  `generate_pairwise_instance(...)` entry point is only a thin adapter over the
  model-specific generators. It compares SK fields and couplings against
  `generate_sk(...)`, EA2D/EA3D fields and sorted couplings against
  `generate_ea(...)`, and RRG fields, couplings, graph type, degree, and graph
  seed metadata against `generate_rrg(...)`.
- `generators/test_pairwise_instance.py` checks the in-memory
  `PairwiseInstance` object. It verifies the documented energy convention
  `H(s)=-sum J_ij s_i s_j - sum h_i s_i`, canonicalizes reversed edges such as
  `(3, 1)` into `(1, 3)`, sorts interactions into a stable order, builds dense
  coupling matrices in either upper-triangular or symmetric form, keeps instance
  hashes stable when only input order or metadata changes, and rejects invalid
  spin assignments outside `{-1, +1}`.
- `generators/test_pairwise_io.py` checks the text file format used for pairwise
  instances. It verifies that existing benchmark files can be loaded with the
  expected model, size, interaction count, and seed; writing and loading an
  instance preserves fields, couplings, and the instance hash; parent output
  directories are created automatically; floating-point values are written with
  the legacy five-decimal formatting; and missing RRG metadata such as graph
  type, degree, and graph seed can be inferred from the file contents.

After generating files under `instances/`, verify the deterministic instance
snapshot with:

```bash
python -m pytest generators/test_reference_instances.py
```

Run the full generator test suite with:

```bash
python -m pytest generators
```
