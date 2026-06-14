#!/usr/bin/env python3
"""Verify experimental framework structure without running experiments."""

from pathlib import Path
import json


def verify_structure():
    """Verify the experimental framework is properly set up."""
    print("=" * 60)
    print("Experimental Framework Structure Verification")
    print("=" * 60)

    # Check framework files
    framework_files = [
        "experiments/src/__init__.py",
        "experiments/src/config.py",
        "experiments/src/results.py",
        "experiments/src/machine_specs.py",
        "experiments/src/runner.py",
        "experiments/src/analysis.py",
        "experiments/src/cli.py",
        "experiments/README.md",
    ]

    print("\n1. Framework Files:")
    all_exist = True
    for file in framework_files:
        path = Path(file)
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {file}")
        if not path.exists():
            all_exist = False

    if all_exist:
        print("  → All framework files present")
    else:
        print("  → Some files missing!")
        return False

    # Check example experiment
    print("\n2. Example Experiment (biazzin/sk_ga_initial):")
    exp_dir = Path("experiments/biazzin/sk_ga_initial")

    # Check experiment_meta.json
    meta_path = exp_dir / "experiment_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  ✓ experiment_meta.json")
        print(f"    - Researcher: {meta['researcher']}")
        print(f"    - Experiment: {meta['experiment_name']}")
        print(f"    - Family: {meta['family']}")
        print(f"    - Algorithms: {', '.join(meta['algorithms'])}")
        print(f"    - Instances: {len(meta['instances'])}")
    else:
        print(f"  ✗ experiment_meta.json missing!")
        return False

    # Check algorithm configs
    for algo in meta['algorithms']:
        algo_config_path = exp_dir / algo / "config.json"
        if algo_config_path.exists():
            with open(algo_config_path) as f:
                algo_config = json.load(f)
            print(f"  ✓ {algo}/config.json")
            print(f"    - Seeds: {algo_config['seeds']}")
            print(f"    - Diagnostics: {algo_config['output_diagnostics']}")
            print(f"    - Parameters: {len(algo_config['solver_parameters'])} keys")
        else:
            print(f"  ✗ {algo}/config.json missing!")
            return False

    # Check SolverResult modification
    print("\n3. Solver Integration:")
    result_file = Path("solvers_v2/src/result.py")
    if result_file.exists():
        content = result_file.read_text()
        if "diagnostics: dict | None = None" in content:
            print("  ✓ SolverResult extended with diagnostics field")
        else:
            print("  ✗ SolverResult not modified")
            return False
    else:
        print("  ✗ SolverResult file not found")
        return False

    print("\n" + "=" * 60)
    print("✓ Framework structure verification PASSED")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install psutil py-cpuinfo")
    print("2. Run example experiment:")
    print("   python -m experiments.src.cli \\")
    print("     experiments/biazzin/sk_ga_initial/experiment_meta.json")
    print("3. See experiments/README.md for documentation")
    return True


if __name__ == "__main__":
    success = verify_structure()
    exit(0 if success else 1)
