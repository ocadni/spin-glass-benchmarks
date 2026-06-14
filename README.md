# Spin Glass Benchmarks

**A standardized repository of benchmark instances for spin glass and combinatorial optimization problems, with reference solver implementations and a reproducible experimental framework.**

## Overview

This repository provides:

1. 📦 **Curated benchmark instances** for multiple problem types (primary focus)
2. 🔧 **Reference solver implementations** demonstrating standard algorithms
3. 📊 **Experimental framework** for reproducible experiments with progress tracking

The goal is to enable fair, reproducible comparisons of optimization algorithms across standardized problem instances.

---

## Benchmark Instances

The repository contains **pre-generated, deterministic benchmark instances** ready for immediate use.

### Available Instance Sets

| Problem Type | Description | Sizes Available | Instances/Size |
|--------------|-------------|-----------------|----------------|
| **SK** | Sherrington-Kirkpatrick (fully connected) | N ∈ {50, 100, 150, 200, 300} | 10 |
| **EA2D** | 2D Edwards-Anderson (square lattice) | N ∈ {100, 256, 1024, 2304} | 10 |
| **EA3D** | 3D Edwards-Anderson (cubic lattice) | N ∈ {512, 1000, 1728, 2744} | 10 |
| **RRG** | Random Regular Graph (k=3) | N ∈ {50, 100, 150, 200, 300} | 10 |
| **XOR-SAT** | k-SAT with XOR clauses | *(generator available)* | — |

**Total:** 190 pre-generated instances across 4 problem types

**Location:** `instances/{sk,ea2d,ea3d,rrg}/`

### Instance File Format

All instances use a standardized text format:

```text
# model=sk N=100 meanJ=0 seed=1051730 distribution=gaussian field=0 num_fields=100 num_couplings=4950
# h_0
0.12345
# h_1
-0.45678
# J_0_1
0.98765
...
```

- **First line**: Metadata (model, size, seed, parameters)
- **Fields**: Local fields `h_i` for each spin
- **Couplings**: Interaction strengths `J_{ij}` for connected spin pairs

**Deterministic generation**: Fixed seeds ensure reproducibility. Same seed + parameters = identical instance.

---

## Problem Models

All models are based on the **Ising spin glass Hamiltonian**:

```
H(s) = -∑_{<i,j>} J_{ij} s_i s_j - ∑_i h_i s_i
```

where `s_i ∈ {-1, +1}` are Ising spins.

### Model Descriptions

**Sherrington-Kirkpatrick (SK)**  
- **Topology**: Fully connected (all-to-all)
- **Couplings**: `J_{ij} ~ N(0, 1/N)`
- **Properties**: Mean-field model, replica symmetry breaking, NP-hard ground state
- **Use case**: Mean-field theory, infinite-dimensional limit

**Edwards-Anderson 2D/3D (EA2D, EA3D)**  
- **Topology**: Nearest-neighbor lattice (2D square or 3D cubic)
- **Couplings**: `J_{ij} ~ N(0, 1)` on lattice edges
- **Properties**: Frustrated interactions, phase transitions, complex landscapes
- **Use case**: Finite-dimensional spin glasses, physical realizations

**Random Regular Graph (RRG)**  
- **Topology**: Random graph with fixed degree k=3
- **Couplings**: `J_{ij} ~ N(0, 1)` on graph edges  
- **Properties**: Sparse, intermediate between lattice and fully connected
- **Use case**: Sparse optimization, constraint satisfaction

**XOR-SAT**  
- **Topology**: k-SAT with XOR constraints
- **Properties**: Planted solution, tunable difficulty
- **Use case**: SAT solving, cryptographic applications
- **Status**: Generator available, instances to be generated

---

## Reference Solvers

The repository includes **reference implementations** of standard algorithms to demonstrate usage and provide baselines.

### Implemented Algorithms

| Algorithm | Family Support | GPU | Description |
|-----------|---------------|-----|-------------|
| **Simulated Annealing (SA)** | SK, EA2D, EA3D | ✓ | Single-trajectory annealing |
| **Population Annealing (PA)** | SK, EA2D, EA3D, RRG | ✓ | Parallel replicas with resampling |
| **Parallel Tempering (PT)** | SK | ✓ | Replica exchange Monte Carlo |
| **Global Annealing (GA)** | SK, EA2D, EA3D | ✓ | **ML-enhanced** with MADE proposals |
| **Greedy** | SK | ✓ | Iterative energy minimization |

**Implementation:** `solvers_v2/` with PyTorch for GPU acceleration

**Features:**
- Live progress tracking with temperature step display
- Reproducible seeding
- Family-specific update schemes (sequential for SK, checkerboard for EA)
- Consistent result format across all solvers

### Example: Running a Solver

```bash
# Setup environment
conda env create -f environments/solvers-core-gpu.yml
conda activate sgbench-solvers-gpu

# Run SK example (N=50, ~30 seconds)
./experiments/biazzin/sk_ga_initial/run_experiment.sh
```

**See:** [Solver documentation](#experimental-framework) below

---

## Quick Start

### 1. Setup Environment

**Check GPU:**
```bash
nvidia-smi  # If available, use GPU environment
```

**Install:**
```bash
# GPU (recommended - 10-100× faster)
conda env create -f environments/solvers-core-gpu.yml
conda activate sgbench-solvers-gpu

# OR CPU-only
conda env create -f environments/solvers-core-cpu.yml
conda activate sgbench-solvers-core
```

**Verify:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 2. Use Instances in Your Code

**Load an instance:**
```python
from solvers_v2.src.io import load_pairwise_couplings

# Load coupling matrix
couplings = load_pairwise_couplings(
    "instances/sk/N100/sk_couplings_N100_J0_seed1051730.txt",
    symmetric=False  # Use symmetric=True for EA models
)

# couplings is a PyTorch tensor ready for your solver
print(f"Instance size: {couplings.shape[0]} spins")
```

**File format is simple text** - easy to parse in any language.

### 3. Run Reference Experiments

**SK baseline example:**
```bash
./experiments/biazzin/sk_ga_initial/run_experiment.sh
```

**EA3D paper reproduction** ([Del Bono et al. PNAS 2025](https://www.pnas.org/doi/full/10.1073/pnas.2534768123)):
```bash
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh
```

---

## Generating Instances

Instances are **pre-generated** and checked into the repository. To generate new ones:

### Setup Generator Environment

```bash
conda env create -f environments/generators.yml
conda activate sgbench-generators
```

### Generate Single Instance

**SK model:**
```bash
python generators/generator.py sk 100 \
  --seed 12345 \
  --outdir instances/sk \
  --meanJ 0 \
  --distribution gaussian \
  --field 0
```

**EA3D model:**
```bash
python generators/generator.py ea3d 1000 \
  --seed 67890 \
  --outdir instances/ea3d \
  --meanJ 0 \
  --distribution gaussian \
  --field 0
```

**XOR-SAT:**
```bash
python generators/generator.py xorsat 100 \
  --seed 11111 \
  --outdir instances/xorsat \
  --k 3 \
  --alpha 0.9
```

### Regenerate All Systematic Instances

```bash
conda activate sgbench-generators
bash scripts/generate_random_seed_systematic.sh
```

**Verify determinism:**
```bash
python -m pytest generators/test_reference_instances.py
```

**See:** [environments/README.md](environments/README.md) for generator details

---

## Experimental Framework

An **optional** framework for running reproducible experiments with multiple solvers and instances.

### Features

- ✅ Batch execution across instances and seeds
- ✅ Live progress tracking with temperature steps
- ✅ Automatic result storage (energy, timing, machine specs)
- ✅ Parameter sweeps
- ✅ SHA256 instance verification
- ✅ Machine specification collection

### Running Experiments

**Using the framework:**
```bash
# Run predefined experiment
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh

# Or directly via CLI
python -m experiments.src.cli experiments/path/to/experiment_meta.json
```

**Live progress display:**
```
[12/100] ⠙ global_annealing | ea3d_couplings_N1000_J0_seed3011732.txt, seed=3141 | 
         Step 15/20 (T=0.156) | Running: 45s | Elapsed: 8m 30s | ETA: 42m | Remaining: 88
```

### Result Storage

Each run creates structured output:
```
experiments/your_experiment/algorithm/runs/N1000_seed3011732_seed3141/
├── common.json          # Energy, runtime, machine specs, parameters
└── diagnostics.json     # Algorithm-specific metrics (optional)
```

### Creating Experiments

**See detailed guide:** [experiments/QUICKSTART.md](experiments/QUICKSTART.md)

**Template:**
```bash
cp experiments/run_experiment_template.sh experiments/your_name/my_experiment/
# Edit configs following examples in experiments/biazzin/
```

### Framework Documentation

- **[experiments/QUICKSTART.md](experiments/QUICKSTART.md)** - Step-by-step experiment creation
- **[experiments/README.md](experiments/README.md)** - Framework overview
- **[EXPERIMENTS_IMPLEMENTATION.md](EXPERIMENTS_IMPLEMENTATION.md)** - Architecture details

---

## Repository Structure

```
spin-glass-benchmarks/
├── instances/                    # 📦 BENCHMARK INSTANCES (main content)
│   ├── sk/                      # Sherrington-Kirkpatrick (190 instances)
│   ├── ea2d/                    # 2D Edwards-Anderson
│   ├── ea3d/                    # 3D Edwards-Anderson  
│   └── rrg/                     # Random Regular Graph
│
├── generators/                   # Instance generation tools
│   ├── generate_sk.py
│   ├── generate_ea.py
│   ├── generate_rrg.py
│   ├── generate_xorsat.py
│   └── generator.py             # Unified CLI
│
├── solvers_v2/                   # 🔧 REFERENCE SOLVER IMPLEMENTATIONS
│   ├── common/                  # Core algorithms (SA, PA, PT, GA)
│   ├── families/                # Model-specific adapters (SK, EA, RRG, XOR-SAT)
│   │   ├── sk/
│   │   ├── ea/
│   │   ├── rrg/
│   │   └── xorsat/
│   └── src/                     # Shared utilities
│
├── experiments/                  # 📊 EXPERIMENTAL FRAMEWORK (optional)
│   ├── src/                     # Framework core (CLI, runner, results)
│   ├── biazzin/                 # Example experiments
│   ├── QUICKSTART.md            # Experiment creation guide
│   └── README.md                # Framework documentation
│
├── environments/                 # Conda environment specs
│   ├── solvers-core-gpu.yml
│   ├── solvers-core-cpu.yml
│   └── generators.yml
│
├── tests/                        # Test suites
└── docs/                         # Additional documentation
```

---

## Example Experiments

The repository includes **working examples** demonstrating the framework:

### SK Model: Global Annealing vs Simulated Annealing

```bash
./experiments/biazzin/sk_ga_initial/run_experiment.sh
```

- **Instances**: SK N=50 (10 instances)
- **Algorithms**: GA, SA
- **Runtime**: ~30 seconds
- **Purpose**: Baseline validation of ML-enhanced solver

### EA3D: Paper Reproduction

```bash
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh
```

- **Paper**: [Del Bono, Ricci-Tersenghi, Zamponi, PNAS 2025](https://www.pnas.org/doi/full/10.1073/pnas.2534768123)
- **Instances**: EA3D N=1000 (10 instances, 5 seeds each)
- **Algorithms**: Global Annealing, Population Annealing
- **Runtime**: ~1.1 hours on Tesla P40 GPU
- **Total runs**: 100 (50 GA + 50 PA)

**Monitor GPU during run:**
```bash
watch -n 2 nvidia-smi
```

---

## Documentation

### Getting Started
- **[README.md](README.md)** (this file) - Overview and quick start
- **[environments/README.md](environments/README.md)** - Environment setup and troubleshooting

### Using the Framework
- **[experiments/QUICKSTART.md](experiments/QUICKSTART.md)** - Create your first experiment
- **[experiments/README.md](experiments/README.md)** - Framework features and usage
- **[EXPERIMENTS_IMPLEMENTATION.md](EXPERIMENTS_IMPLEMENTATION.md)** - Architecture details

### Instance Generation
- **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** - Generator environment guide
- **[generators/README.md](generators/README.md)** - Instance generation details

---

## Testing

### Run Test Suites

```bash
# Full test suite
conda activate sgbench-solvers-gpu
python -m pytest

# Generator tests only
conda activate sgbench-generators
python -m pytest generators/

# Verify instance integrity
python -m pytest generators/test_reference_instances.py
```

### Instance Verification

All instances are verified against deterministic reference snapshots:

```bash
python -m pytest generators/test_reference_instances.py
```

This ensures:
- All expected instances exist
- No unexpected instances
- Byte-level content matches reference
- SHA256 hashes match

---

## Using Instances in Your Own Code

The instances are **simple text files** - use them however you like:

### Python (PyTorch)
```python
from solvers_v2.src.io import load_pairwise_couplings
couplings = load_pairwise_couplings("instances/sk/N100/sk_couplings_N100_J0_seed1051730.txt")
```

### Python (NumPy)
```python
import numpy as np

# Parse manually (simple format)
with open("instances/sk/N100/sk_couplings_N100_J0_seed1051730.txt") as f:
    lines = [l for l in f if not l.startswith("#")]
    # Parse fields and couplings...
```

### Other Languages

The text format is language-agnostic:
1. Skip comment lines (starting with `#`)
2. Read field values
3. Read coupling triplets `(i, j, J_ij)`

**File format documentation**: See [generators/README.md](generators/README.md)

---

## Citation

If you use these benchmark instances in your research, please cite this repository.

**For the Global Annealing solver:**
```bibtex
@article{delbono2025demonstrating,
  title={Demonstrating Real Advantage of Machine-Learning-Enhanced Monte Carlo for Combinatorial Optimization},
  author={Del Bono, Giampaolo and Ricci-Tersenghi, Federico and Zamponi, Francesco},
  journal={Proceedings of the National Academy of Sciences},
  volume={123},
  pages={2534768123},
  year={2025},
  doi={10.1073/pnas.2534768123}
}
```

**arXiv:** [2510.19544v2](https://arxiv.org/abs/2510.19544)

---

## Contributing

Contributions are welcome, especially:
- New benchmark instances (different sizes, models, or distributions)
- Additional solver implementations
- Bug fixes and improvements

**Process:**
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

**For new instances:**
- Follow existing naming conventions
- Include metadata in file header
- Add to systematic generation script if applicable
- Include in test reference snapshot

---

## Quick Command Reference

```bash
# === Setup ===
conda env create -f environments/solvers-core-gpu.yml
conda activate sgbench-solvers-gpu

# === Use Instances ===
# Instances are in: instances/{sk,ea2d,ea3d,rrg}/
# Load in Python: from solvers_v2.src.io import load_pairwise_couplings

# === Run Example Experiments ===
./experiments/biazzin/sk_ga_initial/run_experiment.sh           # SK baseline (~30s)
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh  # EA3D paper (~1.1h)

# === Generate New Instances ===
conda activate sgbench-generators
python generators/generator.py sk 100 --seed 12345 --outdir instances/sk

# === Monitor GPU ===
watch -n 2 nvidia-smi

# === Testing ===
python -m pytest                                     # All tests
python -m pytest generators/test_reference_instances.py  # Verify instances
```

---

## Support

- **Issues**: Report bugs or request features via GitHub Issues  
- **Instances**: All pre-generated instances in `instances/` directory
- **Documentation**: See `experiments/`, `environments/`, and root `*.md` files
- **Examples**: Check `experiments/biazzin/` for working experiments

---

## License

[Add your license here]
