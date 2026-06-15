#!/usr/bin/env python3
"""
Reproduce Figure 3 from arXiv:2510.19544v2
Median success probability over multiple instances comparing PA and GA
for 3D Edwards-Anderson spin glass instances with N=1000 spins
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

# Success threshold: we consider a run successful if it reaches within
# a certain energy gap of the best known energy
# For spin glasses, energies are negative, so we use an additive threshold
SUCCESS_ENERGY_GAP = 0.01  # Within 0.01 energy units of best


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


def extract_instance_seed(instance_path):
    """Extract instance identifier from path."""
    # e.g., instances/ea3d/N1000/ea3d_couplings_N1000_J0_seed3011730.txt
    return Path(instance_path).stem.split('_')[-1]  # Returns "seed3011730"


def compute_best_known_energy(all_results):
    """Compute the best known energy for each instance across all algorithms."""
    best_energies = {}
    for result in all_results:
        instance = result['instance_path']
        best_energy = result['metrics']['best_min_energy']
        if instance not in best_energies:
            best_energies[instance] = best_energy
        else:
            best_energies[instance] = min(best_energies[instance], best_energy)
    return best_energies


def compute_success_probability(results, best_energies, time_points):
    """
    Compute success probability at different time points.

    For each instance and time point, we check how many runs found an energy
    within the success threshold by that time.
    """
    # Group results by instance
    by_instance = defaultdict(list)
    for result in results:
        instance = result['instance_path']
        by_instance[instance].append(result)

    # For each instance, compute success probability at each time point
    instance_probs = []

    for instance, inst_results in by_instance.items():
        best_energy = best_energies[instance]
        # Since energies are negative, best means most negative
        # Success means getting within SUCCESS_ENERGY_GAP (closer to best)
        threshold_energy = best_energy + SUCCESS_ENERGY_GAP

        probs = []
        for t in time_points:
            successes = 0
            for result in inst_results:
                runtime = result['runtime_seconds']
                final_energy = result['metrics']['final_min_energy']

                # Check if this run succeeded by time t
                # Success: energy <= threshold (more negative is better)
                if runtime <= t and final_energy <= threshold_energy:
                    successes += 1

            prob = successes / len(inst_results) if inst_results else 0
            probs.append(prob)

        instance_probs.append(probs)

    # Compute median, 25th and 75th percentiles across instances
    instance_probs = np.array(instance_probs)
    median = np.median(instance_probs, axis=0)
    p25 = np.percentile(instance_probs, 25, axis=0)
    p75 = np.percentile(instance_probs, 25, axis=0)

    return median, p25, p75, len(by_instance)


def main():
    print("Loading GA results...")
    ga_results = load_results(GA_RUNS_DIR)
    print(f"  Found {len(ga_results)} GA runs")

    print("Loading PA results...")
    pa_results = load_results(PA_RUNS_DIR)
    print(f"  Found {len(pa_results)} PA runs")

    all_results = ga_results + pa_results

    print("\nComputing best known energies...")
    best_energies = compute_best_known_energy(all_results)
    print(f"  {len(best_energies)} unique instances")

    # Define time points (logarithmically spaced)
    time_points = np.logspace(-1, 3, 100)  # 0.1s to 1000s

    print("\nComputing success probabilities for GA...")
    ga_median, ga_p25, ga_p75, ga_n_instances = compute_success_probability(
        ga_results, best_energies, time_points
    )

    print(f"Computing success probabilities for PA...")
    pa_median, pa_p25, pa_p75, pa_n_instances = compute_success_probability(
        pa_results, best_energies, time_points
    )

    print(f"\nGA: {ga_n_instances} instances")
    print(f"PA: {pa_n_instances} instances")

    # Create plot similar to Figure 3
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot PA
    ax.plot(time_points, pa_median, 'o-', color='#C85450', linewidth=2,
            markersize=4, label='PA', markevery=5)
    ax.fill_between(time_points, pa_p25, pa_p75, color='#C85450', alpha=0.2)

    # Plot GA
    ax.plot(time_points, ga_median, 's-', color='#4B7BA3', linewidth=2,
            markersize=4, label='GA$_{15}$', markevery=5)
    ax.fill_between(time_points, ga_p25, ga_p75, color='#4B7BA3', alpha=0.2)

    # Add 90% success line
    ax.axhline(y=0.9, color='black', linestyle='--', linewidth=1.5)
    ax.text(time_points[-1] * 0.7, 0.92, '90% success', fontsize=11)

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Success Probability', fontsize=13)
    ax.set_title(f'Success Probability: GA vs PA on 3D EA (N=1000, {ga_n_instances} instances)',
                 fontsize=14, pad=15)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_points[0], time_points[-1])
    ax.set_ylim(0, 1.05)

    # Add info text
    info_text = (f"GA: pop_size=1024, 5 ML moves + 15 local MCS/temp, 20 temps\n"
                 f"PA: pop_size=1024, 10 MCS/temp, 20 temps\n"
                 f"Success: within {SUCCESS_ENERGY_GAP} of best energy")
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save figure
    output_file = EXPERIMENT_DIR / "figure_success_probability_ga_vs_pa.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Also save as PDF
    output_pdf = EXPERIMENT_DIR / "figure_success_probability_ga_vs_pa.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Plot saved to: {output_pdf}")

    plt.show()

    # Print some statistics
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)

    # Find time to 90% success
    ga_t90_idx = np.where(ga_median >= 0.9)[0]
    pa_t90_idx = np.where(pa_median >= 0.9)[0]

    if len(ga_t90_idx) > 0:
        ga_t90 = time_points[ga_t90_idx[0]]
        print(f"GA reaches 90% success at: {ga_t90:.2f} s")
    else:
        print(f"GA does not reach 90% success (max: {ga_median[-1]:.3f})")

    if len(pa_t90_idx) > 0:
        pa_t90 = time_points[pa_t90_idx[0]]
        print(f"PA reaches 90% success at: {pa_t90:.2f} s")
    else:
        print(f"PA does not reach 90% success (max: {pa_median[-1]:.3f})")

    # Runtime statistics
    ga_runtimes = [r['runtime_seconds'] for r in ga_results]
    pa_runtimes = [r['runtime_seconds'] for r in pa_results]

    print(f"\nGA runtime: mean={np.mean(ga_runtimes):.2f}s, "
          f"median={np.median(ga_runtimes):.2f}s, "
          f"std={np.std(ga_runtimes):.2f}s")
    print(f"PA runtime: mean={np.mean(pa_runtimes):.2f}s, "
          f"median={np.median(pa_runtimes):.2f}s, "
          f"std={np.std(pa_runtimes):.2f}s")

    # Energy statistics
    ga_energies = [r['metrics']['final_min_energy'] for r in ga_results]
    pa_energies = [r['metrics']['final_min_energy'] for r in pa_results]

    print(f"\nGA final energy: mean={np.mean(ga_energies):.6f}, "
          f"std={np.std(ga_energies):.6f}")
    print(f"PA final energy: mean={np.mean(pa_energies):.6f}, "
          f"std={np.std(pa_energies):.6f}")


if __name__ == "__main__":
    main()
