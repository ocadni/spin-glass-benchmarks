# Generator Test List

This file summarizes the tests currently implemented in `tests/test_generators.py`.

The distribution tests use these default tolerances:

| Quantity | Tolerance |
| --- | ---: |
| Mean | absolute tolerance `0.05 * expected_std` |
| Standard deviation | relative tolerance `0.05` |
| Third cumulant | absolute tolerance `0.2 * expected_std^3` |
| Fourth cumulant | absolute tolerance `0.4 * expected_std^4` |

## SK Generator

- Checks that an SK instance with `N` spins contains all `N * (N - 1) / 2` complete-graph couplings.
- Checks that all spin fields are written with the requested constant field value.
- Checks that using the same seed gives the same generated instance.
- Checks that Rademacher SK couplings use the SK scaling `1 / sqrt(N)` around the requested mean.
- Checks that SK Gaussian and Rademacher couplings have the expected mean, standard deviation, third cumulant, and fourth cumulant on large instances.

Numerical distribution checks:

| Instance | Distribution | Mean | Standard deviation | Third cumulant | Fourth cumulant |
| --- | --- | ---: | ---: | ---: | ---: |
| `N = 350` | Gaussian | `0.3` | `1 / sqrt(350) = 0.0534522484` | `0` | `0` |
| `N = 350` | Rademacher | `-0.2` | `1 / sqrt(350) = 0.0534522484` | `0` | `-2 / 350^2 = -0.0000163265` |

Numerical tolerances for these SK tests:

| Quantity | Tolerance |
| --- | ---: |
| Mean | `0.0026726124` |
| Standard deviation | relative tolerance `0.05` |
| Third cumulant | `0.0000305960` |
| Fourth cumulant | `0.0000032653` |

## EA Generator

- Checks that 2D EA instances have the expected periodic-lattice number of edges.
- Checks that 3D EA instances have the expected periodic-lattice number of edges.
- Checks that all EA spin fields are written with the requested constant field value.
- Checks that EA couplings are unique and connect valid spin indices.
- Checks that invalid lattice sizes are rejected, for example a non-cube `N` for `dim=3`.
- Checks that Rademacher EA couplings have unit scale around the requested mean.
- Checks that EA Gaussian and Rademacher couplings have the expected mean, standard deviation, third cumulant, and fourth cumulant on large instances.

Numerical distribution checks:

| Instance | Distribution | Mean | Standard deviation | Third cumulant | Fourth cumulant |
| --- | --- | ---: | ---: | ---: | ---: |
| `N = 10000`, `dim = 2` | Gaussian | `-0.2` | `1` | `0` | `0` |
| `N = 10000`, `dim = 2` | Rademacher | `0.5` | `1` | `0` | `-2` |

Numerical tolerances for these EA tests:

| Quantity | Tolerance |
| --- | ---: |
| Mean | `0.05` |
| Standard deviation | relative tolerance `0.05` |
| Third cumulant | `0.2` |
| Fourth cumulant | `0.4` |

## RRG Generator

- Checks that an RRG instance has exactly `N * degree / 2` couplings.
- Checks that every spin has exactly the requested graph degree.
- Checks that invalid degrees are rejected, including odd `N * degree` and `degree >= N`.
- Checks that RRG Gaussian and Rademacher couplings have the expected mean, standard deviation, third cumulant, and fourth cumulant on large instances.

Numerical distribution checks:

| Instance | Distribution | Mean | Standard deviation | Third cumulant | Fourth cumulant |
| --- | --- | ---: | ---: | ---: | ---: |
| `N = 2000`, `degree = 4` | Gaussian | `0.1` | `1` | `0` | `0` |
| `N = 2000`, `degree = 4` | Rademacher | `-0.4` | `1` | `0` | `-2` |

Numerical tolerances for these RRG tests:

| Quantity | Tolerance |
| --- | ---: |
| Mean | `0.05` |
| Standard deviation | relative tolerance `0.05` |
| Third cumulant | `0.2` |
| Fourth cumulant | `0.5` |

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
