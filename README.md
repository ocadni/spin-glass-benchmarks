# Spin Glass Benchmarks

A comprehensive benchmark suite for spin glass optimization algorithms, featuring standardized instances, modern solvers (including ML-enhanced methods), and a complete experimental framework with reproducibility tracking.

## Features

- 🎯 **Pre-generated benchmark instances** for SK, EA2D, EA3D, and RRG models
- 🚀 **Modern solvers**: Simulated Annealing, Population Annealing, Parallel Tempering, Global Annealing (ML-enhanced)
- 📊 **Experiment framework** with live progress tracking, automatic result storage, and machine specs collection
- ⚡ **GPU acceleration** with PyTorch for 10-100× speedup
- 🔬 **Reproducibility**: Deterministic seeds, environment pinning, SHA256 verification

---

## Quick Start (5 minutes)

### 1. Setup Environment

**Check GPU availability:**
```bash
nvidia-smi  # If this works, you have GPU support
```

**Install solver environment:**
```bash
# GPU (recommended - 10-100× faster)
conda env create -f environments/solvers-core-gpu.yml
conda activate sgbench-solvers-gpu

# OR CPU-only
conda env create -f environments/solvers-core-cpu.yml
conda activate sgbench-solvers-core
```

**Verify installation:**
```bash
python -c "import torch; print('✓ PyTorch installed'); print('CUDA available:', torch.cuda.is_available())"
```

### 2. Run Your First Experiment

The repository includes **pre-generated instances** ready to use. Try the example SK experiment:

```bash
# SK model: Global Annealing vs Simulated Annealing (N=50, ~30 seconds)
./experiments/biazzin/sk_ga_initial/run_experiment.sh
```

You'll see live progress like:
```
[1/20] ⠙ global_annealing | sk_couplings_N50_J0_seed1051730.txt, seed=4242 | Step 3/10 (T=0.456) | Running: 2s | ETA: 45s
```

**Results are saved to:** `experiments/biazzin/sk_ga_initial/*/runs/`

### 3. Run the Full Paper Reproduction

**EA3D: Global Annealing vs Population Annealing** (reproduces [Del Bono et al. PNAS 2025](https://www.pnas.org/doi/full/10.1073/pnas.2534768123)):

```bash
# N=1000 (10×10×10 lattice), 100 runs, ~1.1 hours on Tesla P40 GPU
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh
```

This runs:
- 10 disorder realizations × 5 algorithm seeds = 50 runs per algorithm
- Population Annealing: 10 MCS per temperature (~6 minutes total)
- Global Annealing: 5 ML moves + 15 local MCS per temperature (~1 hour total)
- Paper-exact parameters: T=1.92→0.1, 20 logarithmic steps, pop_size=1024

**See the experiment details:** [experiments/biazzin/ea3d_ga_pa_paper_pilot/README.md](experiments/biazzin/ea3d_ga_pa_paper_pilot/README.md)

---

## Available Instances

All instances are **pre-generated** and ready to use in `instances/`:

| Model | Sizes | Instances per Size | Distribution |
|-------|-------|-------------------|--------------|
| **SK** (Sherrington-Kirkpatrick) | N ∈ {50, 100, 150, 200, 300} | 10 | Gaussian, mean=0 |
| **EA2D** (2D Edwards-Anderson) | N ∈ {100, 256, 1024, 2304} | 10 | Gaussian, mean=0 |
| **EA3D** (3D Edwards-Anderson) | N ∈ {512, 1000, 1728, 2744} | 10 | Gaussian, mean=0 |
| **RRG** (Random Regular Graph, k=3) | N ∈ {50, 100, 150, 200, 300} | 10 | Gaussian, mean=0 |

**Instance file format:**
```
# model=sk N=100 meanJ=0 seed=1051730 distribution=gaussian field=0 ...
# h_0
0.12345
# h_1
-0.45678
# J_0_1
0.98765
...
```

---

## Solvers

All solvers support **live progress tracking** showing temperature steps during execution.

### Available Algorithms

| Algorithm | Description | GPU | Use Case |
|-----------|-------------|-----|----------|
| **Simulated Annealing (SA)** | Single-trajectory temperature annealing | ✓ | Baseline, small instances |
| **Population Annealing (PA)** | Parallel replicas with resampling | ✓ | Medium/large instances |
| **Parallel Tempering (PT)** | Replica exchange | ✓ | SK model |
| **Global Annealing (GA)** | **ML-enhanced** with MADE proposal | ✓ | State-of-the-art, research |
| **Greedy** | Iterative energy minimization | ✓ | Quick lower bound |

### Solver Families

- **SK**: All algorithms (SA, PA, PT, GA, Greedy) - fully connected, sequential updates
- **EA2D/EA3D**: SA, PA, GA - lattice models with checkerboard updates
- **RRG**: PA - random regular graphs

**Implementation:** `solvers_v2/`

---

## Creating Your Own Experiments

### Option 1: Use the Template (Recommended)

```bash
# Copy template
cp experiments/run_experiment_template.sh experiments/your_name/my_experiment/run_experiment.sh

# Create configs following the structure in examples
# See experiments/QUICKSTART.md for detailed guide
```

### Option 2: Run Directly

```bash
python -m experiments.src.cli path/to/your/experiment_meta.json
```

**See:** [experiments/QUICKSTART.md](experiments/QUICKSTART.md) for step-by-step instructions

---

## Experiment Framework

The framework handles execution, progress tracking, and result storage:

### Key Features

- ✅ **Live progress display** with temperature steps and ETA
- ✅ **Automatic result storage** (energies, timing, machine specs)
- ✅ **Parameter sweeps** (grid or list)
- ✅ **SHA256 instance verification**
- ✅ **Conda environment detection** with clear error messages
- ✅ **Machine specs collection** (CPU, GPU, OS, conda packages)

### Example Output

```bash
Starting experiment: ea3d_ga_pa_paper_pilot
Researcher: biazzin
Family: ea3d
Algorithms: global_annealing, population_annealing
Instances: 10
Machine: your-hostname
GPU: Tesla P40

global_annealing: 50 runs
  [3/50] ⠙ global_annealing | ea3d_couplings_N1000_J0_seed3011731.txt, seed=3141 | 
         Step 12/20 (T=0.234) | Running: 42s | Elapsed: 2m 15s | ETA: 38m | Remaining: 47
  [3/50] ✓ ea3d_couplings_N1000_J0_seed3011731.txt, seed=3141 (68.3s)
```

### Result Structure

Each run creates:
```
experiments/your_experiment/algorithm_name/runs/N1000_seed3011731_seed3141/
├── common.json          # Energy, runtime, machine specs, parameters
└── diagnostics.json     # Algorithm-specific metrics (optional)
```

**Framework documentation:** [EXPERIMENTS_IMPLEMENTATION.md](EXPERIMENTS_IMPLEMENTATION.md)

---

## Generating New Instances (Optional)

Instances are **already generated**, but you can create new ones:

### Setup Generator Environment

```bash
conda env create -f environments/generators.yml
conda activate sgbench-generators
```

### Generate Single Instance

```bash
python generators/generator.py sk 100 \
  --seed 12345 \
  --outdir instances/sk \
  --meanJ 0 \
  --distribution gaussian \
  --field 0
```

This creates: `instances/sk/N100/sk_couplings_N100_J0_seed12345.txt`

### Regenerate All Systematic Instances

```bash
conda activate sgbench-generators
bash scripts/generate_random_seed_systematic.sh
```

**Verify deterministic generation:**
```bash
python -m pytest generators/test_reference_instances.py
```

**See:** [environments/README.md](environments/README.md) for generator details

---

## Models

### Hamiltonian

All models use the Ising spin glass Hamiltonian:

```
H(s) = -∑_{<i,j>} J_{ij} s_i s_j - ∑_i h_i s_i
```

where `s_i ∈ {-1, +1}` are Ising spins.

### Model Descriptions

**Sherrington-Kirkpatrick (SK)**  
Fully connected mean-field model. Every spin interacts with every other spin. Couplings `J_{ij} ~ N(0, 1/N)`. Ground state is NP-hard to find. Used to study mean-field theory and replica symmetry breaking.

**Edwards-Anderson 2D/3D (EA2D, EA3D)**  
Nearest-neighbor lattice models in 2D or 3D. Couplings `J_{ij} ~ N(0, 1)` on edges. Features frustrated interactions and complex energy landscapes. Used to study finite-dimensional spin glasses and phase transitions.

**Random Regular Graph (RRG)**  
Every node has exactly `k=3` neighbors. Intermediate between finite-dimensional and mean-field models. Couplings `J_{ij} ~ N(0, 1)` on edges. Used to study sparse optimization problems.

---

## Repository Structure

```
spin-glass-benchmarks/
├── environments/          # Conda environment definitions (GPU, CPU, generators)
├── experiments/           # Experiment framework and example experiments
│   ├── src/              # Framework core (CLI, runner, config, results)
│   ├── biazzin/          # Example experiments (SK, EA3D)
│   ├── QUICKSTART.md     # Step-by-step experiment creation guide
│   └── README.md         # Framework documentation
├── instances/            # Pre-generated benchmark instances (ready to use)
│   ├── sk/              # Sherrington-Kirkpatrick instances
│   ├── ea2d/            # 2D Edwards-Anderson instances
│   ├── ea3d/            # 3D Edwards-Anderson instances
│   └── rrg/             # Random Regular Graph instances
├── solvers_v2/          # Solver implementations
│   ├── common/          # Core algorithms (SA, PA, PT, GA)
│   ├── families/        # Model-specific adapters (SK, EA, RRG)
│   └── src/             # Shared utilities (updates, schedules, observables)
├── generators/          # Instance generation scripts
├── tests/               # Test suite
└── docs/                # Additional documentation
```

---

## Documentation

- **[experiments/QUICKSTART.md](experiments/QUICKSTART.md)** - Create your first experiment
- **[experiments/README.md](experiments/README.md)** - Experiment framework overview
- **[environments/README.md](environments/README.md)** - Environment setup and troubleshooting
- **[EXPERIMENTS_IMPLEMENTATION.md](EXPERIMENTS_IMPLEMENTATION.md)** - Framework architecture details
- **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** - Conda environment specifications

---

## Examples

### Quick Examples

```bash
# 1. SK model baseline (GA vs SA, N=50, ~30s)
./experiments/biazzin/sk_ga_initial/run_experiment.sh

# 2. Full paper reproduction (EA3D GA vs PA, N=1000, ~1.1h)
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh
```

### Monitor GPU During Experiment

```bash
# Terminal 1: Run experiment
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh

# Terminal 2: Watch GPU usage
watch -n 2 nvidia-smi
```

### Validate Experiment Before Running

```bash
# Check configs without running solvers
python experiments/biazzin/ea3d_ga_pa_paper_pilot/validate_config.py
```

---

## Testing

### Run Full Test Suite

```bash
# Activate solver environment
conda activate sgbench-solvers-gpu  # or sgbench-solvers-core

# Run all tests
python -m pytest

# Run specific test suites
python -m pytest generators/        # Generator tests
python -m pytest tests/             # Baseline solver tests
```

### Verify Instance Integrity

```bash
conda activate sgbench-generators
python -m pytest generators/test_reference_instances.py
```

This verifies all instances in `instances/` match the deterministic reference snapshot.

---

## Citation

If you use this benchmark suite in your research, please cite:

**For the Global Annealing method:**
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

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## License

[Add your license here]

---

## Support

- **Issues:** Report bugs or request features via GitHub Issues
- **Questions:** See documentation in `experiments/`, `environments/`, and root-level `*.md` files
- **Examples:** Check `experiments/biazzin/` for working examples

---

## Quick Command Reference

```bash
# Setup
conda env create -f environments/solvers-core-gpu.yml
conda activate sgbench-solvers-gpu

# Run example
./experiments/biazzin/sk_ga_initial/run_experiment.sh

# Run paper reproduction
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh

# Monitor GPU
watch -n 2 nvidia-smi

# Validate before running
python experiments/path/to/your/validate_config.py

# Run tests
python -m pytest
```
