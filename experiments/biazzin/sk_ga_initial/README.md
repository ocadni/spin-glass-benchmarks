# SK Global Annealing Initial Experiment

Initial validation experiment comparing global annealing with simulated annealing on small SK instances.

## Quick Run

```bash
./run_experiment.sh
```

This will:
1. Check GPU availability
2. Run global_annealing and simulated_annealing
3. Save results to `*/runs/` directories

## Configuration

- **Instances:** 2 SK instances (N=50)
- **Seeds:** 1729, 4242
- **Algorithms:**
  - `global_annealing` - ML-enhanced with MADE
  - `simulated_annealing` - Standard baseline
- **Total runs:** 8 (2 instances × 2 seeds × 2 algorithms)

## Results

Results are saved in:
```
global_annealing/runs/
└── N50_seed<instance>_seed<seed>/
    ├── common.json         # Metrics, runtime, machine specs
    └── diagnostics.json    # MADE training info (if enabled)

simulated_annealing/runs/
└── N50_seed<instance>_seed<seed>/
    └── common.json
```

## Analyze Results

### Python
```python
from pathlib import Path
from experiments.src.analysis import load_experiment

exp = load_experiment(Path("experiments/biazzin/sk_ga_initial"))

# Get results
ga_results = exp.get_algorithm_results("global_annealing")
sa_results = exp.get_algorithm_results("simulated_annealing")

# Compare
for ga, sa in zip(ga_results, sa_results):
    print(f"Instance: {ga.instance_path}")
    print(f"  GA final energy: {ga.metrics['final_min_energy']:.4f}")
    print(f"  SA final energy: {sa.metrics['final_min_energy']:.4f}")
    print(f"  GA runtime: {ga.runtime_seconds:.2f}s")
    print(f"  SA runtime: {sa.runtime_seconds:.2f}s")
```

### Command line
```bash
# List results
ls global_annealing/runs/

# View one result
cat global_annealing/runs/N50_seed1051730_seed1729/common.json | jq .

# Compare final energies
for dir in global_annealing/runs/*/; do
  echo "$dir:"
  cat "$dir/common.json" | jq -r '.metrics.final_min_energy'
done
```

## Modify Experiment

Edit configuration files:
- `experiment_meta.json` - Change instances, algorithms, tags
- `global_annealing/config.json` - Change GA parameters, seeds
- `simulated_annealing/config.json` - Change SA parameters, seeds

Then run `./run_experiment.sh` again.

## See Also

- [Experiments Framework](../../README.md) - Full documentation
- [Quick Reference](../../QUICKSTART.md) - Command cheat sheet
- [Run Script Template](../../run_experiment_template.sh) - Copy for new experiments
