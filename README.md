# Spin Glass Benchmarks

**Curated benchmark instances + reference solvers for spin glass optimization**

📖 **[Full Documentation](https://ocadni.github.io/spin-glass-benchmarks/)** | 
🚀 **[Quick Start](#quick-start)** | 
📊 **[View Results](https://ocadni.github.io/spin-glass-benchmarks/results.html)**

---

## What's Inside

- **190 benchmark instances** across SK, EA2D, EA3D, RRG families → [Details](https://ocadni.github.io/spin-glass-benchmarks/implementation/instances.html)
- **5 GPU-accelerated solvers**: SA, PA, PT, GA, Greedy → [Coverage matrix](https://ocadni.github.io/spin-glass-benchmarks/implementation/solvers.html)
- **Experimental framework** for reproducible runs → [Guide](experiments/QUICKSTART.md)

**Learn more**: [Problem definitions](https://ocadni.github.io/spin-glass-benchmarks/problem_definition.html) | [Literature](https://ocadni.github.io/spin-glass-benchmarks/literature_references.html) | [Implementation status](https://ocadni.github.io/spin-glass-benchmarks/implementation/status.html)

---

## Quick Start

### 1. Install

```bash
# GPU (recommended - 10-100× faster)
conda env create -f environments/solvers-core-gpu.yml
conda activate sgbench-solvers-gpu

# Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

[Troubleshooting →](environments/README.md) | [CPU-only option →](environments/README.md#cpu-only)

### 2. Use Instances

```python
from solvers_v2.src.io import load_pairwise_couplings

# Load instance
couplings = load_pairwise_couplings(
    "instances/sk/N100/sk_couplings_N100_J0_seed1051730.txt",
    symmetric=False  # Use True for EA/RRG
)

# couplings is a PyTorch tensor: shape (N, N)
print(f"Loaded {couplings.shape[0]} spins")
```

[File format docs →](https://ocadni.github.io/spin-glass-benchmarks/file_format.html)

### 3. Run Example

```bash
# Quick test (~30 seconds)
./experiments/biazzin/sk_ga_initial/run_experiment.sh

# Paper reproduction (~1 hour on GPU)
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh
```

[Create your own experiment →](experiments/QUICKSTART.md)

---

## Available Instances

| Family | Description | Sizes | Count | Location |
|--------|-------------|-------|-------|----------|
| **SK** | Sherrington-Kirkpatrick (fully connected) | 50, 100, 150, 200, 300 | 50 | `instances/sk/` |
| **EA2D** | 2D Edwards-Anderson (square lattice) | 100, 256, 1024, 2304 | 40 | `instances/ea2d/` |
| **EA3D** | 3D Edwards-Anderson (cubic lattice) | 512, 1000, 1728, 2744 | 40 | `instances/ea3d/` |
| **RRG** | Random Regular Graph (k=3) | 50, 100, 150, 200, 300 | 50 | `instances/rrg/` |

**Total**: 180 instances

[Instance library details →](https://ocadni.github.io/spin-glass-benchmarks/implementation/instances.html)

---

## Solvers

| Algorithm | Type | GPU | Families | Status |
|-----------|------|-----|----------|--------|
| **Simulated Annealing** | Single-trajectory | ✓ | All | ✅ |
| **Population Annealing** | Parallel replicas | ✓ | All | ✅ |
| **Parallel Tempering** | Replica exchange | ✓ | SK, RRG | ✅ |
| **Global Annealing** | ML-enhanced (MADE) | ✓ | SK, EA | ✅ |
| **Greedy** | Iterative descent | ✓ | All | ✅ |

**Implementation**: PyTorch-based, GPU-accelerated

[Algorithm details & parameters →](https://ocadni.github.io/spin-glass-benchmarks/implementation/solvers.html)

---

## Running Experiments

The experimental framework provides batch execution, progress tracking, and automatic result storage.

**Create experiment**:
```bash
# Copy template
cp experiments/run_experiment_template.sh experiments/yourname/my_experiment/

# Edit configs (experiment_meta.json, algorithm/config.json)

# Run
./experiments/yourname/my_experiment/run_experiment.sh
```

**Results are saved to**: `experiments/yourname/my_experiment/algorithm/runs/`

[Full guide →](experiments/QUICKSTART.md) | [Framework reference →](experiments/README.md)

---

## Generating Instances

Instances are **pre-generated** and committed to the repository. To create new ones:

```bash
# Setup
conda env create -f environments/generators.yml
conda activate sgbench-generators

# Generate single instance
python generators/generator.py sk 100 --seed 12345 --outdir instances/sk

# Regenerate all systematic instances
bash scripts/generate_random_seed_systematic.sh

# Verify integrity
python -m pytest generators/test_reference_instances.py
```

[Generator documentation →](https://ocadni.github.io/spin-glass-benchmarks/implementation/instances.html#generation-methodology)

---

## Documentation

### Main Documentation Site

📖 **[https://ocadni.github.io/spin-glass-benchmarks/](https://ocadni.github.io/spin-glass-benchmarks/)**

- [Problem Definitions](https://ocadni.github.io/spin-glass-benchmarks/problem_definition.html) - Mathematical formulations, why these problems matter
- [Literature References](https://ocadni.github.io/spin-glass-benchmarks/literature_references.html) - Classical, quantum, ML methods
- [Results](https://ocadni.github.io/spin-glass-benchmarks/results.html) - Benchmark results and analysis
- [Implementation](https://ocadni.github.io/spin-glass-benchmarks/implementation/) - Status, architecture, roadmap

### Technical Guides

- **[experiments/QUICKSTART.md](experiments/QUICKSTART.md)** - Create your first experiment
- **[experiments/README.md](experiments/README.md)** - Framework reference
- **[environments/README.md](environments/README.md)** - Setup troubleshooting
- **[experiments/biazzin/ea3d_ga_pa_paper_pilot/README_PLOTS.md](experiments/biazzin/ea3d_ga_pa_paper_pilot/README_PLOTS.md)** - Example analysis

---

## Repository Structure

```
instances/          # 📦 Benchmark instances (main content)
├── sk/            # Sherrington-Kirkpatrick (50 instances)
├── ea2d/          # 2D Edwards-Anderson (40 instances)
├── ea3d/          # 3D Edwards-Anderson (40 instances)
└── rrg/           # Random Regular Graph (50 instances)

generators/         # Instance generation tools
solvers_v2/         # 🔧 Solver implementations (PyTorch, GPU)
experiments/        # 📊 Experimental framework (optional)
docs/               # 🌐 Documentation site (Quarto)
environments/       # Conda environment specifications
tests/              # Test suites
```

[Architecture details →](https://ocadni.github.io/spin-glass-benchmarks/implementation/architecture.html)

---

## Example: EA3D Paper Reproduction

Reproduce key results from **Del Bono et al., PNAS 2025**:

```bash
# Run experiment (102 runs, ~1 hour on GPU)
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh

# View results
cat experiments/biazzin/ea3d_ga_pa_paper_pilot/README_PLOTS.md
```

**Findings**:
- GA achieves better energy than PA on all 11/11 instances
- GA reaches 90% success rate at ~42s, PA doesn't reach 90% within time limit
- GA shows 1.78× lower variance across seeds (more robust)

[Full analysis →](experiments/biazzin/ea3d_ga_pa_paper_pilot/README_PLOTS.md)

---

## Testing

```bash
# All tests
python -m pytest

# Instance integrity only
python -m pytest generators/test_reference_instances.py

# Solver baseline parity
python -m pytest tests/test_baseline_parity.py
```

---

## Citation

If you use these benchmark instances, please cite:

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

**arXiv**: [2510.19544v2](https://arxiv.org/abs/2510.19544)

---

## Quick Command Reference

```bash
# === Setup ===
conda env create -f environments/solvers-core-gpu.yml
conda activate sgbench-solvers-gpu

# === Run Examples ===
./experiments/biazzin/sk_ga_initial/run_experiment.sh
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh

# === Generate Instances ===
conda activate sgbench-generators
python generators/generator.py sk 100 --seed 12345 --outdir instances/sk

# === Monitor GPU ===
watch -n 2 nvidia-smi

# === Test ===
pytest
```

---

## Contributing

We welcome contributions! See:

- [Roadmap](https://ocadni.github.io/spin-glass-benchmarks/implementation/roadmap.html) - Future priorities
- [Known Issues](https://ocadni.github.io/spin-glass-benchmarks/implementation/known_issues.html) - Bugs and limitations
- [How to contribute](https://ocadni.github.io/spin-glass-benchmarks/implementation/roadmap.html#contributing) - Process

**To contribute**:
1. Fork repository
2. Create feature branch
3. Add tests for changes
4. Submit pull request

---

## Support

- **Documentation**: [https://ocadni.github.io/spin-glass-benchmarks/](https://ocadni.github.io/spin-glass-benchmarks/)
- **Issues**: [GitHub Issues](https://github.com/ocadni/spin-glass-benchmarks/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ocadni/spin-glass-benchmarks/discussions)

---

## License

[Add license here]
