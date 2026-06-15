#!/usr/bin/env python3
"""
Analyze variance across different seeds to determine if GA's robustness
means we need fewer runs per instance compared to PA.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats

# Configuration
EXPERIMENT_DIR = Path(__file__).parent
GA_RUNS_DIR = EXPERIMENT_DIR / "global_annealing" / "runs"
PA_RUNS_DIR = EXPERIMENT_DIR / "population_annealing" / "runs"


def load_results(runs_dir):
    """Load all results from a runs directory."""
    results = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        common_file = run_dir / "common.json"
        if common_file.exists():
            with open(common_file) as f:
                data = json.load(f)
                results.append(data)
    return results


def group_by_instance_and_seed(results):
    """Group results by instance, keeping track of seeds."""
    by_instance = defaultdict(lambda: defaultdict(list))
    for result in results:
        instance = result['instance_path']
        seed = result['seed']
        by_instance[instance][seed].append(result)
    return by_instance


def main():
    print("="*70)
    print("SEED VARIANCE ANALYSIS: Do we need the same number of seeds for GA and PA?")
    print("="*70)

    ga_results = load_results(GA_RUNS_DIR)
    pa_results = load_results(PA_RUNS_DIR)

    # Group by instance and seed
    ga_by_inst = group_by_instance_and_seed(ga_results)
    pa_by_inst = group_by_instance_and_seed(pa_results)

    # Get common instances
    common_instances = sorted(set(ga_by_inst.keys()) & set(pa_by_inst.keys()))
    print(f"\nAnalyzing {len(common_instances)} instances")

    # Analyze variance across seeds for each instance
    ga_within_instance_vars = []
    pa_within_instance_vars = []
    ga_within_instance_stds = []
    pa_within_instance_stds = []

    print("\n" + "="*70)
    print("Per-Instance Analysis: Standard Deviation Across Seeds")
    print("="*70)
    print(f"{'Instance':<45} {'GA Std':>10} {'PA Std':>10} {'Ratio':>8}")
    print("-"*70)

    for instance in common_instances:
        instance_name = Path(instance).stem

        # Get energies for each seed
        ga_energies_by_seed = []
        pa_energies_by_seed = []

        for seed in sorted(ga_by_inst[instance].keys()):
            if seed in ga_by_inst[instance]:
                ga_energies = [r['metrics']['final_min_energy']
                              for r in ga_by_inst[instance][seed]]
                if ga_energies:
                    ga_energies_by_seed.append(np.mean(ga_energies))

        for seed in sorted(pa_by_inst[instance].keys()):
            if seed in pa_by_inst[instance]:
                pa_energies = [r['metrics']['final_min_energy']
                              for r in pa_by_inst[instance][seed]]
                if pa_energies:
                    pa_energies_by_seed.append(np.mean(pa_energies))

        if len(ga_energies_by_seed) > 1 and len(pa_energies_by_seed) > 1:
            ga_std = np.std(ga_energies_by_seed, ddof=1)
            pa_std = np.std(pa_energies_by_seed, ddof=1)

            ga_within_instance_stds.append(ga_std)
            pa_within_instance_stds.append(pa_std)
            ga_within_instance_vars.append(ga_std**2)
            pa_within_instance_vars.append(pa_std**2)

            ratio = pa_std / ga_std if ga_std > 0 else float('inf')
            print(f"{instance_name:<45} {ga_std:>10.6f} {pa_std:>10.6f} {ratio:>8.2f}")

    # Overall statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    ga_mean_std = np.mean(ga_within_instance_stds)
    pa_mean_std = np.mean(pa_within_instance_stds)
    ga_median_std = np.median(ga_within_instance_stds)
    pa_median_std = np.median(pa_within_instance_stds)

    print(f"\nStandard Deviation Across Seeds (within instances):")
    print(f"  GA:  mean={ga_mean_std:.6f}, median={ga_median_std:.6f}")
    print(f"  PA:  mean={pa_mean_std:.6f}, median={pa_median_std:.6f}")
    print(f"  Ratio (PA/GA): {pa_mean_std/ga_mean_std:.2f}x")

    # Statistical test: Are GA variances significantly smaller than PA?
    # Use Levene's test for equality of variances
    _, p_value = stats.levene(ga_within_instance_vars, pa_within_instance_vars)
    print(f"\nLevene's test for equality of variances: p={p_value:.4f}")
    if p_value < 0.05:
        print("  → Variances are significantly different (p < 0.05)")
    else:
        print("  → Cannot reject equal variances hypothesis")

    # Calculate required sample sizes using power analysis
    # For a given effect size and desired power, how many seeds do we need?
    print("\n" + "="*70)
    print("SAMPLE SIZE RECOMMENDATION")
    print("="*70)

    # Calculate coefficient of variation (CV = std/mean)
    ga_all_energies = [r['metrics']['final_min_energy'] for r in ga_results]
    pa_all_energies = [r['metrics']['final_min_energy'] for r in pa_results]

    ga_cv = np.std(ga_all_energies) / abs(np.mean(ga_all_energies))
    pa_cv = np.std(pa_all_energies) / abs(np.mean(pa_all_energies))

    print(f"\nCoefficient of Variation (CV = std/|mean|):")
    print(f"  GA: {ga_cv:.4f}")
    print(f"  PA: {pa_cv:.4f}")
    print(f"  PA has {pa_cv/ga_cv:.2f}x higher relative variability")

    # Estimate required sample size for mean estimation
    # Using formula: n = (z * σ / E)^2
    # where E is desired margin of error
    confidence_level = 0.95
    z_score = 1.96  # for 95% confidence
    desired_margin_pct = 0.01  # 1% of the mean

    ga_margin = desired_margin_pct * abs(np.mean(ga_all_energies))
    pa_margin = desired_margin_pct * abs(np.mean(pa_all_energies))

    ga_n_required = (z_score * ga_mean_std / ga_margin) ** 2
    pa_n_required = (z_score * pa_mean_std / pa_margin) ** 2

    print(f"\nRequired seeds per instance (95% confidence, ±1% margin):")
    print(f"  GA: {ga_n_required:.1f} seeds")
    print(f"  PA: {pa_n_required:.1f} seeds")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of standard deviations
    ax1 = axes[0, 0]
    bins = np.linspace(0, max(max(ga_within_instance_stds),
                             max(pa_within_instance_stds)), 20)
    ax1.hist(ga_within_instance_stds, bins=bins, alpha=0.6,
             label='GA', color='#4B7BA3', edgecolor='black')
    ax1.hist(pa_within_instance_stds, bins=bins, alpha=0.6,
             label='PA', color='#C85450', edgecolor='black')
    ax1.axvline(ga_mean_std, color='#4B7BA3', linestyle='--',
                linewidth=2, label=f'GA mean: {ga_mean_std:.4f}')
    ax1.axvline(pa_mean_std, color='#C85450', linestyle='--',
                linewidth=2, label=f'PA mean: {pa_mean_std:.4f}')
    ax1.set_xlabel('Standard Deviation Across Seeds', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Distribution of Within-Instance Variability', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Scatter plot: GA std vs PA std
    ax2 = axes[0, 1]
    ax2.scatter(pa_within_instance_stds, ga_within_instance_stds,
                s=100, alpha=0.6, color='purple', edgecolors='black')
    max_std = max(max(ga_within_instance_stds), max(pa_within_instance_stds))
    ax2.plot([0, max_std], [0, max_std], 'k--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('PA Std Dev', fontsize=11)
    ax2.set_ylabel('GA Std Dev', fontsize=11)
    ax2.set_title('Instance-wise Variability: GA vs PA', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add text showing how many points are below diagonal
    below_diag = sum(1 for ga, pa in zip(ga_within_instance_stds,
                                          pa_within_instance_stds) if ga < pa)
    total = len(ga_within_instance_stds)
    ax2.text(0.05, 0.95, f'GA more stable:\n{below_diag}/{total} instances',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Box plot comparison
    ax3 = axes[1, 0]
    bp = ax3.boxplot([ga_within_instance_stds, pa_within_instance_stds],
                      patch_artist=True, widths=0.6)
    ax3.set_xticklabels(['GA', 'PA'])
    colors = ['#4B7BA3', '#C85450']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel('Standard Deviation', fontsize=11)
    ax3.set_title('Variability Comparison', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Required sample size visualization
    ax4 = axes[1, 1]
    margins = [0.005, 0.01, 0.02, 0.05, 0.10]  # Different margin requirements
    ga_samples = []
    pa_samples = []

    for margin_pct in margins:
        ga_margin = margin_pct * abs(np.mean(ga_all_energies))
        pa_margin = margin_pct * abs(np.mean(pa_all_energies))
        ga_samples.append((z_score * ga_mean_std / ga_margin) ** 2)
        pa_samples.append((z_score * pa_mean_std / pa_margin) ** 2)

    x = np.arange(len(margins))
    width = 0.35
    ax4.bar(x - width/2, ga_samples, width, label='GA',
            color='#4B7BA3', alpha=0.7)
    ax4.bar(x + width/2, pa_samples, width, label='PA',
            color='#C85450', alpha=0.7)
    ax4.set_xlabel('Desired Margin of Error (% of mean)', fontsize=11)
    ax4.set_ylabel('Required Number of Seeds', fontsize=11)
    ax4.set_title('Sample Size Requirements (95% confidence)', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{m*100:.1f}%' for m in margins])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')

    plt.tight_layout()

    output_file = EXPERIMENT_DIR / "figure_seed_variance_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    output_pdf = EXPERIMENT_DIR / "figure_seed_variance_analysis.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Plot saved to: {output_pdf}")

    plt.show()

    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    print(f"\nBased on the analysis:")
    print(f"  - GA shows {pa_mean_std/ga_mean_std:.2f}x LOWER seed-to-seed variability")
    print(f"  - GA is more stable on {below_diag}/{total} instances")

    if pa_n_required > ga_n_required * 1.5:
        print(f"\n✓ YES, different seed counts are justified:")
        print(f"  - GA needs ~{int(np.ceil(ga_n_required))} seeds per instance")
        print(f"  - PA needs ~{int(np.ceil(pa_n_required))} seeds per instance")
        print(f"  - This is a {pa_n_required/ga_n_required:.1f}x difference!")
        print(f"\n  Current setup (5 seeds each) is:")
        print(f"    - {'Adequate' if ga_n_required <= 5 else 'Borderline'} for GA")
        print(f"    - {'Adequate' if pa_n_required <= 5 else 'Insufficient'} for PA")
    else:
        print(f"\n✓ Current approach (same seeds for both) is reasonable")
        print(f"  - Difference in requirements is small (<1.5x)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
