# Environment Setup Implementation Summary

## What Was Implemented

A comprehensive conda environment structure with CPU and GPU support for spin glass benchmarking.

## New Structure

```
environments/
├── README.md                    # Complete setup guide
├── generators.yml               # For instance generation (pinned versions)
├── solvers-core-cpu.yml        # For experiments (CPU only)
└── solvers-core-gpu.yml        # For experiments (GPU with conda CUDA)
```

## Key Features

### 1. Three Environments

**sgbench-generators**
- Purpose: Generate benchmark instances
- Python 3.12.2 + NumPy 2.2.3 (pinned for RNG reproducibility)
- Minimal dependencies

**sgbench-solvers-core** (CPU)
- Purpose: Run experiments on CPU
- PyTorch 2.7.0 (CPU-only)
- All experiment framework dependencies

**sgbench-solvers-gpu** (GPU) - NEW
- Purpose: Run experiments on GPU (10-100x faster)
- PyTorch 2.7.0 with CUDA 12.1 (conda-managed)
- No system CUDA installation required
- All experiment framework dependencies

### 2. Conda-Managed CUDA

GPU environment uses conda to install CUDA toolkit:
- No system CUDA needed
- Automatic version compatibility
- Isolated per environment
- Works on any system with NVIDIA GPU

### 3. Experiment Framework Support

Added to solver environments:
- `psutil` - For machine specs collection
- `py-cpuinfo` - For better CPU information

### 4. Backward Compatibility

Symlinks maintain existing paths:
- `generators/environment.yml` → `environments/generators.yml`
- `solvers_v2/envs/core-cpu.yml` → `environments/solvers-core-cpu.yml`

## Documentation Updates

### Main README.md
Added "Quick Start" section:
1. Check GPU availability (`nvidia-smi`)
2. Install environment (GPU or CPU)
3. Verify installation
4. Run example experiment

Updated "Repository Structure" to highlight key directories.

### environments/README.md (NEW)
Comprehensive guide with:
- First-time setup (3 steps)
- Detailed environment descriptions
- Typical workflow examples
- Extensive troubleshooting section
- FAQ

### experiments/README.md
Added "Prerequisites" section referencing environment setup.

### .gitignore
Added `*.lock` to ignore conda lock files.

## First-Time User Workflow

```bash
# 1. Check hardware
nvidia-smi

# 2. Install environment
conda env create -f environments/solvers-core-gpu.yml  # or -cpu.yml
conda activate sgbench-solvers-gpu

# 3. Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 4. Run experiment
python -m experiments.src.cli experiments/biazzin/sk_ga_initial/experiment_meta.json
```

## Files Created

**New:**
- `environments/README.md`
- `environments/generators.yml`
- `environments/solvers-core-cpu.yml`
- `environments/solvers-core-gpu.yml`
- `ENVIRONMENT_SETUP.md` (this file)

**Modified:**
- `README.md` (Quick Start + Repository Structure)
- `experiments/README.md` (Prerequisites)
- `.gitignore` (*.lock)

**Symlinks:**
- `generators/environment.yml` → `../environments/generators.yml`
- `solvers_v2/envs/core-cpu.yml` → `../../environments/solvers-core-cpu.yml`

**Unchanged:**
- `solvers_v2/envs/core-cpu-linux-64.lock` (kept for reference)

## Benefits

**For New Users:**
- Clear path from clone to running experiments
- Single decision point: GPU or CPU?
- Comprehensive troubleshooting

**For GPU Users:**
- 10-100x speedup
- No system CUDA hassle
- Easy to verify

**For Existing Users:**
- Backward compatible
- Experiment framework fully supported
- Optional upgrade to GPU

**For Maintainers:**
- Central environment definitions
- Clear documentation
- Easy to update

## Next Steps for Users

1. **Install environment:**
   ```bash
   conda env create -f environments/solvers-core-gpu.yml  # or -cpu
   ```

2. **See detailed docs:**
   - [environments/README.md](environments/README.md) - Complete environment guide
   - [experiments/QUICKSTART.md](experiments/QUICKSTART.md) - Experiment creation
   - [experiments/README.md](experiments/README.md) - Full experiment docs

3. **Optional:** Install generator environment if creating new instances
   ```bash
   conda env create -f environments/generators.yml
   ```
