# Quick Start: Reproducing Paper Plots

This guide shows how to quickly reproduce the plots from arXiv:2510.19544v2.

## Prerequisites

Make sure you have matplotlib and scipy installed in your solver environment:

```bash
~/conda-envs/sgbench-solvers-gpu/bin/pip install matplotlib scipy
```

## Generate All Plots

Run all three analysis scripts:

```bash
cd experiments/biazzin/ea3d_ga_pa_paper_pilot

# 1. Success probability comparison (reproduces paper Figure 3)
~/conda-envs/sgbench-solvers-gpu/bin/python plot_success_probability.py

# 2. Energy and runtime comparison (reproduces elements from Figures 2, 4)
~/conda-envs/sgbench-solvers-gpu/bin/python plot_energy_comparison.py

# 3. Seed variance analysis (NEW - shows GA needs fewer seeds!)
~/conda-envs/sgbench-solvers-gpu/bin/python analyze_seed_variance.py
```

## Generated Files

Each script creates both PNG (300 DPI) and PDF versions:

1. **`figure_success_probability_ga_vs_pa.{png,pdf}`**
   - Shows GA reaches 90% success, PA plateaus at 60%
   - Demonstrates GA's superior robustness

2. **`figure_energy_comparison_ga_vs_pa.{png,pdf}`**
   - 4-panel comparison: scatter, runtime, energy distribution, quality-time tradeoff
   - Shows GA wins on all 11/11 instances
   - PA is 20× faster but achieves worse quality

3. **`figure_seed_variance_analysis.{png,pdf}`**
   - NEW: Proves GA is more robust statistically
   - Shows GA needs 1.78× fewer seeds than PA
   - Justifies using different numbers of seeds per algorithm

## Key Results Summary

```
Metric                    | GA          | PA          | Winner
--------------------------|-------------|-------------|--------
Mean final energy         | -1.7008     | -1.6935     | GA ✓
Best energy found         | -1.7440     | -1.7360     | GA ✓
Mean runtime              | 37.1s       | 1.8s        | PA ✓
90% success rate          | Yes (~42s)  | No (<60%)   | GA ✓
Wins per instance         | 11/11       | 0/11        | GA ✓
Seed-to-seed variability  | 0.0018      | 0.0031      | GA ✓ (1.78× lower)
Required seeds (95% CI)   | 1-2         | 3-5         | GA ✓
```

## Your Question: Different Seed Counts?

**YES!** You are absolutely correct. The statistical analysis shows:

- **GA is more robust** → needs fewer seeds per instance (1-2 vs 3-5)
- **PA is less stable** → needs more seeds for reliable statistics
- **Current setup (5 seeds each)** is actually conservative for GA

**Recommendation for future experiments**:
- GA: 2-3 seeds per instance is sufficient
- PA: 4-5 seeds per instance to account for higher variability

This is a **double advantage** of GA:
1. Better solution quality
2. More consistent results (fewer runs needed)

## Documentation

See [README_PLOTS.md](README_PLOTS.md) for complete documentation including:
- Detailed experimental setup
- Comparison with paper methodology  
- Statistical analysis of results
- Interpretation of findings
