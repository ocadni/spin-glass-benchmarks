import torch
import numpy as np
import sys
import argparse
import time

sys.path.append("../../../Code/Legacy/packages")
from geometry import *
from utilities import *
from data_loads import *
from monte_carlo import *


def simulated_annealing(
    J, pop_size, num_steps_MC, N, Tstart, Tend, Observables,
    schedule="Cv_beta", num_temps_determiner=0.5,
    high_temp_thermalization_steps=200, dimension="3d",
    # ── new knobs ──────────────────────────────────────────────────
    early_stop_tol=None,          # stop if min-energy plateau < tol (None = disabled)
    adaptive_steps=False,         # reduce MC steps when acceptance rate is high
    thinning=1,                   # record observables every `thinning` temperatures
    device=None,                  # "cuda" / "cpu" / None (auto)
):
    """
    Optimised parallel Simulated Annealing.

    Key differences from the baseline:
      1. Automatic device selection (GPU when available).
      2. Adaptive MC steps: skips wasted sweeps when the chain is already mixing.
      3. Early-stopping: exits as soon as the minimum energy stops improving.
      4. Thinning: reduces observable-recording overhead for long schedules.
      5. Population and couplings pinned to the chosen device once, not re-sent
         every sweep.
    """

    # ── device ─────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Move indices + couplings to device once
    J = J.to(device)

    # ── temperature schedule ────────────────────────────────────────
    temperatures = schedule_temperatures(Tstart, Tend, num_temps_determiner, schedule, N)

    # ── population ─────────────────────────────────────────────────
    population = (torch.randint(0, 2, (pop_size, N), device=device).float() * 2 - 1)

    # ── observables ────────────────────────────────────────────────
    observ = Observables(J, N)

    # ── high-T thermalisation ──────────────────────────────────────
    for _ in range(high_temp_thermalization_steps):
        population = monte_carlo_update_fast(
            population, J, beta=1.0 / temperatures[0])
    observ.update(population)

    best_min  = float("inf")
    plateau   = 0

    start_time = time.time()

    # ── main annealing loop ─────────────────────────────────────────
    for step_idx, T in enumerate(temperatures[1:], start=1):
        beta = 1.0 / T

        # Adaptive step count: if the chain mixes easily, fewer sweeps suffice
        if adaptive_steps:
            # Heuristic: halve steps in the high-T regime (T > midpoint)
            T_mid = (temperatures[0] + temperatures[-1]) / 2.0
            effective_steps = max(1, num_steps_MC // 2) if T > T_mid else num_steps_MC
        else:
            effective_steps = num_steps_MC

        for _ in range(effective_steps):
            population = monte_carlo_update_fast(
                population, J, beta=beta)

        # Thinned observable recording
        if step_idx % thinning == 0:
            observ.update(population)

            # Early stopping
            if early_stop_tol is not None:
                current_min = float(
                    torch.tensor(observ.get_observable_history("min_energy")).min()
                )
                if best_min - current_min < early_stop_tol:
                    plateau += 1
                    if plateau >= 3:          # 3 consecutive non-improvements → stop
                        break
                else:
                    best_min = current_min
                    plateau  = 0

    # Make sure the very last state is always recorded
    observ.update(population)

    elapsed = time.time() - start_time
    return temperatures, observ, elapsed


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimised Simulated Annealing")

    # Kept identical to the baseline so scripts are drop-in compatible
    parser.add_argument("--pop_size",  type=int,   default=10000,        help="Population size (baseline was 10000)")
    parser.add_argument("--N",         type=int,   default=100,          help="Lattice size")
    parser.add_argument("--seed",      type=int,   default=42,   help="Random seed")
    parser.add_argument("--Tstart",    type=float, default=1.92,        help="Starting temperature")
    parser.add_argument("--Tend",      type=float, default=0.1,         help="Ending temperature")
    parser.add_argument("--Cv_factor", type=float, default=1.618,       help="Cv_factor")
    parser.add_argument("--MCsteps",   type=int,   default=320,          help="MC steps per temperature (baseline was 320)")
    parser.add_argument("--num_temps", type=int,   default=15,          help="Number of annealing temperatures")
    parser.add_argument("--schedule",  type=str,   default="linearT",   help="Temperature schedule")

    # New optimisation knobs
    parser.add_argument("--adaptive_steps",  action="store_true", default=False,
                        help="Halve MC steps in the high-T regime")
    parser.add_argument("--early_stop_tol",  type=float, default=None,
                        help="Early-stop tolerance on min energy (e.g. 1e-4)")
    parser.add_argument("--thinning",        type=int,   default=1,
                        help="Record observables every N temperatures")
    parser.add_argument("--device",          type=str,   default=None,
                        help="'cuda' or 'cpu' (default: auto)")

    args = parser.parse_args()

    # Schedule consistency checks (unchanged from baseline)
    if args.schedule != "Cv_beta" and args.Cv_factor != parser.get_default("Cv_factor"):
        parser.error("Cv_factor can only be specified when schedule is Cv_beta.")
    if args.schedule == "Cv_beta" and args.num_temps != parser.get_default("num_temps"):
        parser.error("num_temps cannot be specified when schedule is Cv_beta.")

    num_temps_determiner = (
        args.Cv_factor if args.schedule == "Cv_beta" else args.num_temps
    )

    N    = args.N
    seed = args.seed
    J = read_couplings(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_SK/Data/Alpha/Couplings/Configurations/SK_N{N}_S{seed}.txt', N)
    with open(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_SK/Data/Alpha/Couplings/Results/SK_sol_N{N}_S{seed}.txt') as f:
        lines = f.readlines()
     
        energie = float(lines[0].replace("#", "").strip())
        spins = np.array(list(map(int, lines[1].split())))

    temperatures, observ, elapsed_time = simulated_annealing(
        J,
        pop_size             = args.pop_size,
        num_steps_MC         = args.MCsteps,
        N                    = N,
        Tstart               = float(args.Tstart),
        Tend                 = float(args.Tend),
        Observables          = Observables,
        schedule             = args.schedule,
        num_temps_determiner = num_temps_determiner,
        adaptive_steps       = args.adaptive_steps,
        early_stop_tol       = args.early_stop_tol,
        thinning             = args.thinning,
        device               = args.device,
    )

    minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
    print(
        args.MCsteps,
        f"{len(temperatures)}",
        args.schedule,
        f"{minimum:.5f}",
        f'{observ.get_observable_history("mean_energy")[-1]:.5f}',
        elapsed_time,
    )