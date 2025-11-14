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
    python generators/generate_sk.py > benchmarks/sk_N100.txt
    ```

2.  **Run a solver:**
    ```bash
    # (Assuming solver is adapted to read from file)
    python solvers/sa.py benchmarks/sk_N100.txt
    ```
