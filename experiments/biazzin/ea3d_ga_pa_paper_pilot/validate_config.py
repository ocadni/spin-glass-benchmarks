#!/usr/bin/env python
"""Validate experiment configuration without running solver workloads.

This script checks:
1. JSON configs are valid and schema-compliant
2. Instance files exist
3. Algorithm names are valid
4. Solver parameters are structurally sound
"""

import json
import sys
from pathlib import Path


def validate_json_file(path: Path) -> dict:
    """Load and validate JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {path}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        sys.exit(1)


def validate_experiment_meta(meta: dict, base_dir: Path) -> None:
    """Validate experiment_meta.json structure."""
    required_fields = ["schema_version", "researcher", "experiment_name", "family", "instances", "algorithms"]
    for field in required_fields:
        if field not in meta:
            print(f"❌ Missing required field: {field}")
            sys.exit(1)

    # Check instances exist
    repo_root = base_dir.parent.parent.parent
    for instance in meta["instances"]:
        instance_path = repo_root / instance
        if not instance_path.exists():
            print(f"❌ Instance file not found: {instance}")
            sys.exit(1)
    print(f"✓ All {len(meta['instances'])} instance(s) exist")

    # Check algorithms
    valid_algorithms = ["simulated_annealing", "population_annealing", "global_annealing", "parallel_tempering"]
    for algo in meta["algorithms"]:
        if algo not in valid_algorithms:
            print(f"⚠ Warning: Unexpected algorithm: {algo}")
    print(f"✓ Algorithms: {', '.join(meta['algorithms'])}")


def validate_solver_config(config: dict, algo_name: str) -> None:
    """Validate solver config.json structure."""
    required_fields = ["schema_version", "algorithm", "seeds", "solver_parameters"]
    for field in required_fields:
        if field not in config:
            print(f"❌ Missing required field in {algo_name} config: {field}")
            sys.exit(1)

    params = config["solver_parameters"]
    common_params = ["L", "pop_size", "Tstart", "Tend", "num_temps", "schedule", "high_temp_thermalization_steps"]
    for param in common_params:
        if param not in params:
            print(f"⚠ Warning: Missing common parameter in {algo_name}: {param}")

    # Check algorithm-specific parameters
    if config["algorithm"] == "global_annealing":
        if "MLMCsteps" not in params or "swap_step" not in params:
            print(f"❌ GA missing MLMCsteps or swap_step")
            sys.exit(1)
        print(f"✓ GA config: {params['MLMCsteps']} ML steps × {params['swap_step']} local MCS")
    elif config["algorithm"] == "population_annealing":
        if "MCsteps" not in params:
            print(f"❌ PA missing MCsteps")
            sys.exit(1)
        print(f"✓ PA config: {params['MCsteps']} MCS per temperature")


def main():
    script_dir = Path(__file__).parent
    print(f"Validating experiment: {script_dir.name}\n")

    # Load and validate experiment_meta.json
    meta_path = script_dir / "experiment_meta.json"
    print("Checking experiment_meta.json...")
    meta = validate_json_file(meta_path)
    validate_experiment_meta(meta, script_dir)

    # Load and validate algorithm configs
    print("\nChecking algorithm configs...")
    for algo in meta["algorithms"]:
        config_path = script_dir / algo / "config.json"
        if not config_path.exists():
            print(f"❌ Config not found: {config_path}")
            sys.exit(1)
        config = validate_json_file(config_path)
        validate_solver_config(config, algo)

    # Verify paper settings
    print("\nVerifying paper settings...")
    for algo in meta["algorithms"]:
        config_path = script_dir / algo / "config.json"
        config = validate_json_file(config_path)
        params = config["solver_parameters"]

        # Check temperature range
        if params["Tstart"] != 1.92 or params["Tend"] != 0.1:
            print(f"⚠ {algo}: non-paper temperature range (Tstart={params['Tstart']}, Tend={params['Tend']})")

        # Check schedule
        if params["schedule"] != "logT":
            print(f"⚠ {algo}: non-paper schedule ({params['schedule']})")

        # Check thermalization
        if params["high_temp_thermalization_steps"] != 200:
            print(f"⚠ {algo}: non-paper thermalization ({params['high_temp_thermalization_steps']} MCS)")

    print("\n✓ Configuration validation passed")
    print(f"\nThis is a PILOT experiment with:")
    print(f"  - Small population ({config['solver_parameters']['pop_size']})")
    print(f"  - Short temperature ladder ({config['solver_parameters']['num_temps']} temps)")
    print(f"  - Single instance (N=512)")
    print(f"\nFor full paper-scale runs, increase pop_size, num_temps, and use N1000/N2744 instances.")


if __name__ == "__main__":
    main()
