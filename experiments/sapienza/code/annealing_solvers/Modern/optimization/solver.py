"""Command-line wrapper for the Modern annealing solvers.

Example:
    python solver.py /path/to/ea3d_couplings_N1000_J0_seed418527.txt sa \
        --population-size 256 --num-steps-mc 10 --num-temps 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


# These modules retain absolute-style imports from the legacy package.  Make
# the wrapper runnable from any working directory, not only this directory.
OPTIMIZATION_DIR = Path(__file__).resolve().parent
LEGACY_PACKAGES_DIR = OPTIMIZATION_DIR.parents[1] / "Legacy" / "packages"
if str(LEGACY_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_PACKAGES_DIR))

from global_annealing import global_annealing
from monte_carlo import Observables, get_num_spins, read_couplings
from population_annealing import population_annealing
from simulated_annealing import simulated_annealing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Modern annealing solver on a benchmark instance."
    )
    parser.add_argument("instance", type=Path, help="Path to a benchmark instance file")
    parser.add_argument(
        "annealer",
        choices=("sa", "pa", "ga"),
        help="Annealing scheme to run",
    )

    common = parser.add_argument_group("common annealing parameters")
    common.add_argument("--population-size", type=int, default=256)
    common.add_argument("--num-steps-mc", type=int, default=10,
                        help="Local MC sweeps at each temperature")
    common.add_argument("--t-start", type=float, default=3.0)
    common.add_argument("--t-end", type=float, default=0.2)
    common.add_argument(
        "--schedule",
        choices=("linearT", "linearBeta", "logT"),
        default="linearBeta",
    )
    common.add_argument("--num-temps", type=int, default=100)
    common.add_argument("--thermalization-steps", type=int, default=200)
    common.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="auto prefers CUDA, then Apple MPS, then CPU",
    )
    common.add_argument("--seed", type=int, default=None)

    population = parser.add_argument_group("population annealing parameters")
    population.add_argument(
        "--reweight-mode", choices=("multinomial", "systematic"), default="multinomial"
    )

    global_options = parser.add_argument_group("global annealing parameters")
    global_options.add_argument("--swap-step", type=int, default=1)
    global_options.add_argument("--batch-size", type=int, default=256)
    global_options.add_argument("--num-epochs-start", type=int, default=40)
    global_options.add_argument("--num-epochs-retrain", type=int, default=1)

    parser.add_argument("--json", action="store_true", help="Print the result summary as JSON")
    args = parser.parse_args()

    if not args.instance.is_file():
        parser.error(f"instance file does not exist: {args.instance}")
    for option in ("population_size", "num_steps_mc", "num_temps", "thermalization_steps"):
        if getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if args.t_start <= 0 or args.t_end <= 0:
        parser.error("--t-start and --t-end must be positive")
    if args.annealer == "ga":
        for option in ("swap_step", "batch_size", "num_epochs_start", "num_epochs_retrain"):
            if getattr(args, option) <= 0:
                parser.error(f"--{option.replace('_', '-')} must be positive")
    return args


def select_device(requested: str) -> torch.device:
    """Resolve a CLI device request, checking backend availability."""
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not mps_available:
        raise RuntimeError("Apple MPS was requested but is not available")
    return torch.device(requested)


def run_solver(args: argparse.Namespace) -> dict[str, object]:
    """Load one instance, run the requested annealer, and return a summary."""
    device = select_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    J, fields = read_couplings(args.instance, return_fields=True, device=device)

    common_args = dict(
        J=J,
        pop_size=args.population_size,
        num_steps_MC=args.num_steps_mc,
        Tstart=args.t_start,
        Tend=args.t_end,
        Observables=Observables,
        schedule=args.schedule,
        num_temps=args.num_temps,
        high_temp_thermalization_steps=args.thermalization_steps,
        fields=fields,
    )
    if args.annealer == "sa":
        temperatures, observables, elapsed = simulated_annealing(**common_args)
        timings = {"elapsed_seconds": elapsed}
    elif args.annealer == "pa":
        temperatures, observables, elapsed = population_annealing(
            **common_args, reweight_mode=args.reweight_mode
        )
        timings = {"elapsed_seconds": elapsed}
    else:
        temperatures, observables, train_elapsed, annealing_elapsed = global_annealing(
            **common_args,
            swap_step=args.swap_step,
            batch_size=args.batch_size,
            num_epochs_start=args.num_epochs_start,
            num_epochs_retrain=args.num_epochs_retrain,
        )
        timings = {
            "training_seconds": train_elapsed,
            "annealing_seconds": annealing_elapsed,
            "elapsed_seconds": train_elapsed + annealing_elapsed,
        }

    history = observables.observables
    return {
        "annealer": args.annealer,
        "instance": str(args.instance),
        "device": str(device),
        "num_spins": get_num_spins(J),
        "has_external_fields": fields is not None,
        "num_temperatures": len(temperatures),
        "t_start": float(temperatures[0]),
        "t_end": float(temperatures[-1]),
        "final_min_energy_per_spin": history["min_energy"][-1],
        "best_min_energy_per_spin": min(history["min_energy"]),
        **timings,
    }


def print_summary(summary: dict[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"annealer: {summary['annealer']}")
    print(f"instance: {summary['instance']}")
    print(f"device: {summary['device']}")
    print(f"spins: {summary['num_spins']}")
    print(f"external fields: {'nonzero' if summary['has_external_fields'] else 'zero'}")
    print(f"temperatures: {summary['num_temperatures']} ({summary['t_start']} -> {summary['t_end']})")
    print(f"final minimum energy/spin: {summary['final_min_energy_per_spin']:.8f}")
    print(f"best minimum energy/spin: {summary['best_min_energy_per_spin']:.8f}")
    if "training_seconds" in summary:
        print(f"training time: {summary['training_seconds']:.3f} s")
        print(f"annealing time: {summary['annealing_seconds']:.3f} s")
    print(f"total elapsed time: {summary['elapsed_seconds']:.3f} s")


def main() -> None:
    args = parse_args()
    try:
        summary = run_solver(args)
    except (RuntimeError, ValueError, NotImplementedError) as error:
        raise SystemExit(f"solver error: {error}") from error
    print_summary(summary, args.json)


if __name__ == "__main__":
    main()
