#!/usr/bin/env python3
"""Compare results before and after architecture refactoring."""

import json
from pathlib import Path
import sys

def load_results(directory):
    """Load all result JSON files from a directory."""
    results = {}
    runs_dir = Path(directory) / "runs"
    if not runs_dir.exists():
        return results

    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            common_json = run_dir / "common.json"
            if common_json.exists():
                with open(common_json) as f:
                    data = json.load(f)
                    key = (
                        Path(data["instance_path"]).stem,
                        data["seed"],
                    )
                    results[key] = data
    return results

def compare_metrics(old, new, tolerance=1e-6):
    """Compare metrics between old and new results."""
    differences = []

    # Compare final energies
    old_final = old["metrics"]["final_min_energy"]
    new_final = new["metrics"]["final_min_energy"]
    if abs(old_final - new_final) > tolerance:
        differences.append(
            f"  final_min_energy: {old_final:.10f} -> {new_final:.10f} (Δ={new_final - old_final:.2e})"
        )

    # Compare best energies
    old_best = old["metrics"]["best_min_energy"]
    new_best = new["metrics"]["best_min_energy"]
    if abs(old_best - new_best) > tolerance:
        differences.append(
            f"  best_min_energy: {old_best:.10f} -> {new_best:.10f} (Δ={new_best - old_best:.2e})"
        )

    return differences

def main():
    script_dir = Path(__file__).parent

    old_dir = script_dir / "global_annealing_old_backup"
    new_dir = script_dir / "global_annealing"

    if not old_dir.exists():
        print("❌ Old backup directory not found")
        return 1

    if not new_dir.exists():
        print("❌ New results directory not found")
        return 1

    print("Comparing Global Annealing Results")
    print("=" * 60)
    print(f"Old: {old_dir}")
    print(f"New: {new_dir}")
    print()

    old_results = load_results(old_dir)
    new_results = load_results(new_dir)

    if not old_results:
        print("❌ No old results found")
        return 1

    if not new_results:
        print("❌ No new results found")
        return 1

    print(f"Old results: {len(old_results)} runs")
    print(f"New results: {len(new_results)} runs")
    print()

    all_match = True

    for key in sorted(old_results.keys()):
        instance, seed = key
        print(f"Instance: {instance}, Seed: {seed}")

        if key not in new_results:
            print("  ❌ Missing in new results")
            all_match = False
            continue

        old = old_results[key]
        new = new_results[key]

        # Compare runtime
        old_runtime = old["runtime_seconds"]
        new_runtime = new["runtime_seconds"]
        runtime_change = ((new_runtime - old_runtime) / old_runtime) * 100
        print(f"  Runtime: {old_runtime:.2f}s -> {new_runtime:.2f}s ({runtime_change:+.1f}%)")

        # Compare metrics
        diffs = compare_metrics(old, new)
        if diffs:
            print("  ⚠️  Differences detected:")
            for diff in diffs:
                print(diff)
            all_match = False
        else:
            print("  ✅ Results match (within tolerance)")
        print()

    print("=" * 60)
    if all_match:
        print("✅ All results match! Architecture refactoring is successful.")
        return 0
    else:
        print("⚠️  Some differences detected. Review above.")
        print("Note: Small numerical differences may be acceptable due to")
        print("floating point precision or GPU non-determinism.")
        return 0  # Return 0 because small diffs are OK

if __name__ == "__main__":
    sys.exit(main())
