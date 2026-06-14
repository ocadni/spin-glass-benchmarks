# EA3D GA vs PA Paper Pilot

Pilot reproduction of the 3D Edwards-Anderson benchmark from Del Bono,
Ricci-Tersenghi, and Zamponi, "Demonstrating Real Advantage of
Machine-Learning-Enhanced Monte Carlo for Combinatorial Optimization"
(`arXiv:2510.19544v2`, PNAS `10.1073/pnas.2534768123`).

## Scope

**📊 FULL PAPER-SCALE REPRODUCTION**

This experiment uses the exact parameters from the paper:
- **Population**: 1024 (as per paper)
- **Temperature ladder**: 20 logarithmic steps from T=1.92 to T=0.1
- **Instance size**: N=1000 (10×10×10 3D lattice)
- **Instances**: 10 independent disorder realizations
- **Seeds**: 5 algorithm seeds per instance
- **Total runs**: 2 algorithms × 10 instances × 5 seeds = **100 runs**

### Algorithm Parameters (from paper)
- **PA**: 10 MCS per temperature step
- **GA**: 5 global moves per temperature, 15 local MCS per global move
- **Both**: 200 high-temperature thermalization MCS
- **GA only**: 40 initial MADE training epochs, 1 retrain epoch per temperature

### Expected Runtime
Based on pilot (N=512, 6 temps):
- PA: ~0.6s × (1000/512)² × (20/6) ≈ **7s per run** → ~6 minutes total for PA
- GA: ~6.6s × (1000/512)² × (20/6) ≈ **70s per run** → ~1 hour total for GA

**Total estimated time: ~1.1 hours** (GPU-accelerated on Tesla P40)

## Paper Settings Used

- 3D Edwards-Anderson spin glass without external fields
- `Tstart = 1.92`
- `Tend = 0.1`
- logarithmically spaced temperatures
- 200 high-temperature thermalization MCS
- PA: 10 local MCS per temperature
- GA: 5 global moves per temperature, 15 local MCS per global move

The pilot uses a smaller population and shorter temperature ladder to check
runtime before scaling out.

## Validation

Before running, validate the configuration:

```bash
python experiments/biazzin/ea3d_ga_pa_paper_pilot/validate_config.py
```

This checks JSON validity, instance file existence, and verifies paper settings
(Tstart, Tend, schedule, thermalization, algorithm ratios) without launching
solver workloads.

## Run

**IMPORTANT**: Requires the solver conda environment (`sgbench-solvers-gpu` or
`sgbench-solvers-core`). The script will error if the environment is missing
unless you explicitly set `ALLOW_BASE_PYTHON=1`.

```bash
./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh
```

Or directly:

```bash
python -m experiments.src.cli experiments/biazzin/ea3d_ga_pa_paper_pilot/experiment_meta.json
```

To use current Python (not recommended):

```bash
ALLOW_BASE_PYTHON=1 ./experiments/biazzin/ea3d_ga_pa_paper_pilot/run_experiment.sh
```
