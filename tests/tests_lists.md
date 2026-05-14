# Generator Test List

This file summarizes the tests currently implemented in `tests/test_generators.py`.

## SK Generator

- Checks that an SK instance with `N` spins contains all `N * (N - 1) / 2` complete-graph couplings.
- Checks that all spin fields are written with the requested constant field value.
- Checks that using the same seed gives the same generated instance.
- Checks that Rademacher SK couplings use the SK scaling `1 / sqrt(N)` around the requested mean.

## EA Generator

- Checks that 2D EA instances have the expected periodic-lattice number of edges.
- Checks that 3D EA instances have the expected periodic-lattice number of edges.
- Checks that all EA spin fields are written with the requested constant field value.
- Checks that EA couplings are unique and connect valid spin indices.
- Checks that invalid lattice sizes are rejected, for example a non-cube `N` for `dim=3`.
- Checks that Rademacher EA couplings have unit scale around the requested mean.

## RRG Generator

- Checks that an RRG instance has exactly `N * degree / 2` couplings.
- Checks that every spin has exactly the requested graph degree.
- Checks that invalid degrees are rejected, including odd `N * degree` and `degree >= N`.

## Command-Line Generator

- Checks that EA output files include the dimension in the model name, for example `ea2d`.
- Checks that the first header line contains the expected model, `N`, mean coupling, and seed metadata.
- Checks that field lines are written after the header.
- Checks that model-specific arguments fail on the wrong models:
  - `--L` is rejected for SK.
  - `--dim` is rejected for RRG.
  - `--degree` is rejected for EA.

## How to Run

From the repository root:

```bash
python -m pytest tests
```
