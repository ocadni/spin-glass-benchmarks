#!/usr/bin/env python3
"""
Create energy comparison plots for GA vs PA
Similar to analysis in arXiv:2510.19544v2
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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


def group_by_instance(results):
    """Group results by instance."""
    by_instance = defaultdict(list)
    for result in results:
        instance = result['instance_path']
        by_instance[instance].append(result)
    return by_instance


def main():
    print("Loading results...")
    ga_results = load_results(GA_RUNS_DIR)
    pa_results = load_results(PA_RUNS_DIR)

    print(f"GA runs: {len(ga_results)}")
    print(f"PA runs: {len(pa_results)}")

    # Group by instance
    ga_by_inst = group_by_instance(ga_results)
    pa_by_inst = group_by_instance(pa_results)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Scatter plot: GA vs PA best energies per instance
    ax1 = fig.add_subplot(gs[0, 0])

    ga_best_per_inst = []
    pa_best_per_inst = []
    instances = sorted(set(ga_by_inst.keys()) & set(pa_by_inst.keys()))

    for instance in instances:
        ga_energies = [r['metrics']['final_min_energy'] for r in ga_by_inst[instance]]
        pa_energies = [r['metrics']['final_min_energy'] for r in pa_by_inst[instance]]

        ga_best_per_inst.append(np.min(ga_energies))
        pa_best_per_inst.append(np.min(pa_energies))

    ax1.scatter(pa_best_per_inst, ga_best_per_inst, s=100, alpha=0.7,
                color='#4B7BA3', edgecolors='black', linewidth=1)

    # Add diagonal line
    min_val = min(min(ga_best_per_inst), min(pa_best_per_inst))
    max_val = max(max(ga_best_per_inst), max(pa_best_per_inst))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5)

    ax1.set_xlabel('PA Best Energy', fontsize=12)
    ax1.set_ylabel('GA Best Energy', fontsize=12)
    ax1.set_title('Best Energy: GA vs PA (per instance)', fontsize=13, pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Count how many times GA wins
    ga_wins = sum(1 for ga, pa in zip(ga_best_per_inst, pa_best_per_inst) if ga < pa)
    ax1.text(0.05, 0.95, f'GA wins: {ga_wins}/{len(instances)}',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Runtime comparison
    ax2 = fig.add_subplot(gs[0, 1])

    ga_runtimes = [r['runtime_seconds'] for r in ga_results]
    pa_runtimes = [r['runtime_seconds'] for r in pa_results]

    bp_data = [pa_runtimes, ga_runtimes]
    bp = ax2.boxplot(bp_data, patch_artist=True, widths=0.6)
    ax2.set_xticklabels(['PA', 'GA'])

    colors = ['#C85450', '#4B7BA3']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.set_title('Runtime Comparison', fontsize=13, pad=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add mean annotations
    for i, (data, color) in enumerate(zip(bp_data, colors), 1):
        mean_val = np.mean(data)
        ax2.plot(i, mean_val, 'D', color='darkred', markersize=8, zorder=3)
        ax2.text(i, mean_val, f' {mean_val:.1f}s',
                verticalalignment='bottom', fontsize=10, fontweight='bold')

    # 3. Energy distribution histogram
    ax3 = fig.add_subplot(gs[1, 0])

    ga_all_energies = [r['metrics']['final_min_energy'] for r in ga_results]
    pa_all_energies = [r['metrics']['final_min_energy'] for r in pa_results]

    bins = np.linspace(
        min(min(ga_all_energies), min(pa_all_energies)),
        max(max(ga_all_energies), max(pa_all_energies)),
        30
    )

    ax3.hist(pa_all_energies, bins=bins, alpha=0.6, label='PA',
             color='#C85450', edgecolor='black', linewidth=0.5)
    ax3.hist(ga_all_energies, bins=bins, alpha=0.6, label='GA',
             color='#4B7BA3', edgecolor='black', linewidth=0.5)

    ax3.axvline(np.mean(pa_all_energies), color='#C85450',
                linestyle='--', linewidth=2, label=f'PA mean: {np.mean(pa_all_energies):.4f}')
    ax3.axvline(np.mean(ga_all_energies), color='#4B7BA3',
                linestyle='--', linewidth=2, label=f'GA mean: {np.mean(ga_all_energies):.4f}')

    ax3.set_xlabel('Final Minimum Energy', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Energy Distribution (all runs)', fontsize=13, pad=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Energy vs Runtime scatter
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.scatter(pa_runtimes, pa_all_energies, s=50, alpha=0.5,
                label='PA', color='#C85450', edgecolors='black', linewidths=0.5)
    ax4.scatter(ga_runtimes, ga_all_energies, s=50, alpha=0.5,
                label='GA', color='#4B7BA3', edgecolors='black', linewidths=0.5)

    ax4.set_xlabel('Runtime (seconds)', fontsize=12)
    ax4.set_ylabel('Final Minimum Energy', fontsize=12)
    ax4.set_title('Energy vs Runtime', fontsize=13, pad=10)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('GA vs PA Performance Comparison: 3D EA N=1000',
                 fontsize=15, fontweight='bold', y=0.995)

    # Save figure
    output_file = EXPERIMENT_DIR / "figure_energy_comparison_ga_vs_pa.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    output_pdf = EXPERIMENT_DIR / "figure_energy_comparison_ga_vs_pa.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Plot saved to: {output_pdf}")

    plt.show()

    # Print statistics
    print("\n" + "="*70)
    print("STATISTICS SUMMARY")
    print("="*70)

    print(f"\nNumber of instances: {len(instances)}")
    print(f"Runs per algorithm: GA={len(ga_results)}, PA={len(pa_results)}")

    print(f"\n--- ENERGY STATISTICS ---")
    print(f"GA: mean={np.mean(ga_all_energies):.6f}, "
          f"std={np.std(ga_all_energies):.6f}, "
          f"best={np.min(ga_all_energies):.6f}")
    print(f"PA: mean={np.mean(pa_all_energies):.6f}, "
          f"std={np.std(pa_all_energies):.6f}, "
          f"best={np.min(pa_all_energies):.6f}")

    energy_diff = np.mean(ga_all_energies) - np.mean(pa_all_energies)
    print(f"\nMean energy difference (GA - PA): {energy_diff:.6f}")
    print(f"  → GA is {'better' if energy_diff < 0 else 'worse'} by "
          f"{abs(energy_diff):.6f} energy units")

    print(f"\n--- RUNTIME STATISTICS ---")
    print(f"GA: mean={np.mean(ga_runtimes):.2f}s, "
          f"median={np.median(ga_runtimes):.2f}s, "
          f"std={np.std(ga_runtimes):.2f}s")
    print(f"PA: mean={np.mean(pa_runtimes):.2f}s, "
          f"median={np.median(pa_runtimes):.2f}s, "
          f"std={np.std(pa_runtimes):.2f}s")

    speedup = np.mean(ga_runtimes) / np.mean(pa_runtimes)
    print(f"\nPA speedup factor: {speedup:.1f}x faster than GA")

    print(f"\n--- INSTANCE-WISE COMPARISON ---")
    print(f"GA achieves better energy on {ga_wins}/{len(instances)} instances")
    print(f"Average per-instance energy difference: "
          f"{np.mean([ga - pa for ga, pa in zip(ga_best_per_inst, pa_best_per_inst)]):.6f}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
