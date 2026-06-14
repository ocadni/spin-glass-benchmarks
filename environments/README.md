# Conda Environments

This repository uses separate conda environments for different tasks to maintain reproducibility and avoid dependency conflicts.

## First-Time Setup

### Step 1: Check Your Hardware

Do you have an NVIDIA GPU?
```bash
nvidia-smi
```

- ✅ **Command works?** → Use GPU environment (10-100x faster for solvers)
- ❌ **Command fails?** → Use CPU environment

### Step 2: Install Environment

**For GPU users (recommended):**
```bash
conda env create -f environments/solvers-core-gpu.yml
conda activate sgbench-solvers-gpu
```

**For CPU-only users:**
```bash
conda env create -f environments/solvers-core-cpu.yml
conda activate sgbench-solvers-core
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output (GPU):**
```
PyTorch: 2.7.0
CUDA available: True
```

**Expected output (CPU):**
```
PyTorch: 2.7.0
CUDA available: False
```

### Step 4: (Optional) Install Generator Environment

Only needed if you plan to generate new benchmark instances:
```bash
conda env create -f environments/generators.yml
conda activate sgbench-generators
```

## Available Environments

### 1. `solvers-core-gpu.yml` → `sgbench-solvers-gpu` (Recommended)

**Use for:** Running experiments and solvers on GPU

**Speed:** 10-100x faster than CPU for most solvers

**Includes:** 
- PyTorch 2.7.0 with CUDA 12.1 (conda-managed, no system CUDA needed)
- NumPy 2.2.3, pandas, networkx, tqdm
- Experiment framework dependencies (psutil, py-cpuinfo)
- pytest for testing

**Requirements:** 
- NVIDIA GPU with compute capability ≥ 3.5
- No system CUDA installation needed (conda manages it)

**Activate:**
```bash
conda activate sgbench-solvers-gpu
```

**Verify GPU:**
```bash
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not available\"}')"
```

### 2. `solvers-core-cpu.yml` → `sgbench-solvers-core`

**Use for:** Running experiments and solvers on CPU only (no GPU)

**Includes:**
- PyTorch 2.7.0 (CPU-only build)
- NumPy 2.2.3, pandas, networkx, tqdm
- Experiment framework dependencies (psutil, py-cpuinfo)
- pytest for testing

**When to use:**
- No GPU available
- Testing on CPU
- Small experiments where GPU isn't needed

**Activate:**
```bash
conda activate sgbench-solvers-core
```

### 3. `generators.yml` → `sgbench-generators`

**Use for:** Generating new benchmark instances

**Why separate?** 
- Pins Python 3.12.2 and NumPy 2.2.3 for RNG reproducibility
- Generated instances must be bit-for-bit reproducible
- Isolated from solver dependencies

**Includes:**
- Python 3.12.2 (pinned)
- NumPy 2.2.3 (pinned)
- pytest

**Activate:**
```bash
conda activate sgbench-generators
```

**Verify versions:**
```bash
python -c "import sys, numpy as np; print(f'Python: {sys.version.split()[0]}'); print(f'NumPy: {np.__version__}')"
```

Expected output:
```
Python: 3.12.2
NumPy: 2.2.3
```

## Typical Workflow

### One-Time Setup

```bash
# Install solver environment (choose GPU or CPU)
conda env create -f environments/solvers-core-gpu.yml  # or -cpu.yml

# Optional: install generator environment (only if generating instances)
conda env create -f environments/generators.yml
```

### Generate Instances (Optional)

```bash
conda activate sgbench-generators
python generators/generate_sk.py --N 100 --num 10 --seed 42
```

### Run Experiments (Main Workflow)

```bash
conda activate sgbench-solvers-gpu  # or sgbench-solvers-core
python -m experiments.src.cli experiments/yourname/my_experiment/experiment_meta.json
```

### Analyze Results

```bash
conda activate sgbench-solvers-gpu  # or sgbench-solvers-core
python
>>> from pathlib import Path
>>> from experiments.src.analysis import load_experiment
>>> exp = load_experiment(Path("experiments/yourname/my_experiment"))
>>> results = exp.get_algorithm_results("simulated_annealing")
>>> for r in results:
...     print(f"{r.instance_path}: {r.metrics['final_min_energy']:.4f}")
```

## Updating Environments

If environment files are updated in the repository:

```bash
# Update existing environment
conda env update -f environments/solvers-core-gpu.yml --prune

# Or remove and recreate
conda env remove -n sgbench-solvers-gpu
conda env create -f environments/solvers-core-gpu.yml
```

## Troubleshooting

### "CUDA out of memory" Error

**Solution 1:** Reduce batch size or population size in your experiment config:
```json
{
  "solver_parameters": {
    "pop_size": 50,        // Reduce from 100
    "batch_size": 128      // Reduce from 256
  }
}
```

**Solution 2:** Switch to CPU environment:
```bash
conda activate sgbench-solvers-core
```

### "nvidia-smi: command not found"

**Cause:** No NVIDIA GPU or drivers not installed

**Solution:** Use CPU environment:
```bash
conda env create -f environments/solvers-core-cpu.yml
conda activate sgbench-solvers-core
```

### "CUDA device capability insufficient"

**Cause:** GPU is too old (compute capability < 3.5)

**Solution:** Use CPU environment or upgrade GPU

### Slow Performance on GPU

**Check GPU is actually being used:**
```bash
watch -n 1 nvidia-smi
```

Run experiment and verify GPU utilization > 0%

**If GPU not used:** Check PyTorch installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`. If `False`, reinstall GPU environment.

### System CUDA vs Conda CUDA

**Q:** I have CUDA installed system-wide. Will there be conflicts?

**A:** No. Our GPU environment uses conda-managed CUDA, which is completely isolated. Your system CUDA won't interfere.

**Q:** Can I use my system CUDA instead?

**A:** Not recommended. Conda-managed CUDA ensures version compatibility and is easier to manage. If you must use system CUDA, modify `solvers-core-gpu.yml` to remove the `pytorch-cuda` dependency.

### Import Errors

**"ModuleNotFoundError: No module named 'psutil'"**

**Cause:** Using old environment without experiment framework dependencies

**Solution:** Update or recreate environment:
```bash
conda env update -f environments/solvers-core-gpu.yml --prune
```

### Generator Version Mismatch

**"Generated instances don't match reference"**

**Cause:** Wrong Python or NumPy version in generator environment

**Verify versions:**
```bash
conda activate sgbench-generators
python -c "import sys, numpy as np; print(sys.version.split()[0], np.__version__)"
```

Must be exactly: `3.12.2 2.2.3`

**Solution:** Recreate generator environment:
```bash
conda env remove -n sgbench-generators
conda env create -f environments/generators.yml
```

## Environment Comparison

| Feature | generators | solvers-core-cpu | solvers-core-gpu |
|---------|-----------|------------------|------------------|
| **Purpose** | Generate instances | Run experiments (CPU) | Run experiments (GPU) |
| **Python** | 3.12.2 (pinned) | 3.12.2 | 3.12.2 |
| **NumPy** | 2.2.3 (pinned) | 2.2.3 | 2.2.3 |
| **PyTorch** | ❌ | 2.7.0 (CPU) | 2.7.0 (GPU) |
| **CUDA** | ❌ | ❌ | 12.1 (conda-managed) |
| **GPU Support** | ❌ | ❌ | ✅ |
| **Experiments** | ❌ | ✅ | ✅ |
| **Speed** | N/A | 1x | 10-100x |

## Best Practices

1. **Use GPU environment for experiments** - Much faster, no downside
2. **Keep generator environment separate** - Ensures reproducibility
3. **Don't mix environments** - Always activate before running commands
4. **Update regularly** - When environment files change in repo
5. **Verify after installation** - Run verification commands above

## FAQ

**Q: Which environment should I use most of the time?**

A: `sgbench-solvers-gpu` (or `-core` if no GPU). This is your main environment for running experiments.

**Q: Do I need all three environments?**

A: No. Most users only need the solver environment (GPU or CPU). Only install `generators` if you're generating new instances.

**Q: Can I have both CPU and GPU environments?**

A: Yes! They're separate environments with different names. You can switch between them:
```bash
conda activate sgbench-solvers-gpu  # Use GPU
conda activate sgbench-solvers-core # Use CPU
```

**Q: How much disk space do these environments use?**

A: 
- generators: ~500 MB
- solvers-core-cpu: ~2 GB
- solvers-core-gpu: ~4 GB (includes CUDA toolkit)

**Q: Will this work on Mac/Windows?**

A: 
- **Mac (Apple Silicon)**: Use CPU environment (no CUDA support)
- **Mac (Intel)**: Use CPU environment
- **Windows**: GPU and CPU both work
- **Linux**: GPU and CPU both work (recommended)

## See Also

- [Experiments Quick Reference](../experiments/QUICKSTART.md) - Create and run experiments
- [Main README](../README.md) - Repository overview
- [Experiment Framework](../experiments/README.md) - Detailed experiment documentation
