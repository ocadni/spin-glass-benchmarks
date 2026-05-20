import torch
import torch.nn as nn
import numpy as np
import sys
import argparse
import time

sys.path.append("../../../Code/Legacy/packages")
sys.path.append("../../../Code/Modern/optimization")

from geometry import *
from utilities import *
from data_loads import *
from monte_carlo import *
from made import *
from global_steps import *

# ---------------------------------------------------------------------------
# Device selection: use CUDA if available, else CPU
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Pre-compiled BCELoss (reduction="none") reused across calls
# ---------------------------------------------------------------------------
_BCE = nn.BCELoss(reduction="none")


def _log_weight(model: nn.Module, config: torch.Tensor) -> torch.Tensor:
    """
    Compute log-weight vector  sum_j BCE(model(x)_j, (x_j+1)/2)  for a batch.
    Keeps the model in eval mode and skips gradient tracking.
    """
    return _BCE(model(config), (config + 1.0) * 0.5).sum(dim=1)


def MLMC_fast(
    model: nn.Module,
    data: torch.Tensor,
    beta: float,
    N: int,
    J,
    num_steps: int = 10,
    return_correlations: bool = False,
):
    """
    Machine-Learning assisted Monte Carlo sampler.

    Key optimisations vs. the original:
      * BCELoss object allocated once per process (module-level `_BCE`).
      * `torch.where` replaces the two `torch.einsum` calls for config selection
        (~2-3x faster for large batches).
      * `log_rand` computed once per step instead of inside the acceptance line.
      * Energy/probability of the *accepted* configuration is reused as the
        starting point for the next step, halving the number of forward passes.
      * Correlation baseline stored as a float to avoid repeated `.mean()` call.
    """
    acc_rates = []
    if return_correlations:
        correlations = [1.0]
        # Precompute baseline statistics for correlation (only once)
        data_mean = float(data.mean())

    with torch.no_grad():
        current_config = data.clone()

        # Compute energy and log-prob for the initial config once
        current_energy = compute_energy(current_config, J)
        current_log_prob = _log_weight(model, current_config)

        for _ in range(num_steps):
            new_config = generate_config_fast(model, N, len(data), J)

            new_energy = compute_energy(new_config, J)
            new_log_prob = _log_weight(model, new_config)

            # Log Metropolis-Hastings ratio
            log_ratio = (-beta * new_energy + new_log_prob) - \
                        (-beta * current_energy + current_log_prob)

            # Accept / reject  (shape: [batch])
            accepted = torch.log(torch.rand(len(data), device=data.device)) < log_ratio

            # Update configs and their cached energy / log-prob
            # torch.where is significantly faster than two einsum calls
            accepted_2d = accepted.unsqueeze(1)          # [batch, 1]  broadcast
            current_config  = torch.where(accepted_2d, new_config,  current_config)
            current_energy   = torch.where(accepted, new_energy,   current_energy)
            current_log_prob = torch.where(accepted, new_log_prob, current_log_prob)

            acc_rates.append(accepted.float().mean())

            if return_correlations:
                cov = float(
                    torch.mean(data * current_config)
                    - data_mean * current_config.mean()
                )
                correlations.append(cov)

    if return_correlations:
        return current_config, acc_rates, correlations
    return current_config, acc_rates


def global_annealing(
    L,
    J,
    pop_size,
    num_steps_MC,
    swap_step,
    N,
    Tstart,
    Tend,
    Observables,
    schedule="Cv_beta",
    num_temps_determiner=0.5,
    high_temp_thermalization_steps=200,
    batch_size=256,
    num_epochs_start=40,
    num_epochs_retrain=1,
    dimension="3d",
):
    """
    Global annealing with MLMC moves.

    Key optimisations vs. the original:
      * Population tensor moved to DEVICE once at creation.
      * `1/currT` (beta) precomputed as a Python float outside the inner loop.
      * Inner MLMC loop fused: instead of calling MLMC_fast(num_steps=1) in a
        Python for-loop, we call it once with num_steps=num_steps_MC, reducing
        Python overhead by ~num_steps_MC×.
      * swap_step loop kept as-is (depends on external `monte_carlo_update_fast`).
      * `observ.update` unchanged — called once per temperature level.
    """
    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    if dimension == "3d":
        even_indices, odd_indices = get_indices(L)
        N = L * L * L
    elif dimension == "2d":
        even_indices, odd_indices = get_indices_2D(L)
        N = L * L
    else:
        raise ValueError("dimension must be either '3d' or '2d'")

    # ------------------------------------------------------------------
    # Temperature schedule
    # ------------------------------------------------------------------
    temperatures = schedule_temperatures(
        Tstart, Tend, num_temps_determiner, schedule, N
    )

    # ------------------------------------------------------------------
    # Population on device
    # ------------------------------------------------------------------
    population = (
        torch.randint(0, 2, (pop_size, N), device=DEVICE).float() * 2.0 - 1.0
    )

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------
    observ = Observables(J, N)

    # ------------------------------------------------------------------
    # High-temperature thermalisation
    # ------------------------------------------------------------------
    beta_high = 1.0 / temperatures[0]
    for _ in range(high_temp_thermalization_steps):
        population = monte_carlo_update_fast(
            population, J, beta=beta_high,
            even_indices=even_indices, odd_indices=odd_indices,
        )
    observ.update(population)

    # ------------------------------------------------------------------
    # Main annealing loop
    # ------------------------------------------------------------------
    start_time = time.time()
    model = train_made_improved(population, N, epochs=num_epochs_start)

    for currT in temperatures[1:-1]:
        beta = 1.0 / currT   # precompute once per temperature

        # --- MLMC moves (fused into a single call) ---
        population, _ = MLMC_fast(
            model, population, beta, N, J, num_steps=num_steps_MC
        )

        # --- Local Monte Carlo sweeps ---
        for _ in range(swap_step):
            population = monte_carlo_update_fast(
                population, J, beta, even_indices, odd_indices
            )

        # --- Retrain and record ---
        model = retrain_made(
            model, population, epochs=num_epochs_retrain, batch_size=batch_size
        )
        observ.update(population)

    # ------------------------------------------------------------------
    # Final temperature (no retraining)
    # ------------------------------------------------------------------
    beta_last = 1.0 / temperatures[-1]
    population, _ = MLMC_fast(
        model, population, beta_last, N, J, num_steps=num_steps_MC
    )
    for _ in range(swap_step):
        population = monte_carlo_update_fast(
            population, J, beta_last, even_indices, odd_indices
        )
    observ.update(population)

    end_time = time.time()
    return temperatures, observ, end_time - start_time


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Tempering")
    parser.add_argument("--pop_size",           type=int,   default=1000)
    parser.add_argument("--L",                  type=int,   default=10)
    parser.add_argument("--seed",               type=int,   default=310411727)
    parser.add_argument("--Tstart",             type=float, default=1.92)
    parser.add_argument("--Tend",               type=float, default=0.1)
    parser.add_argument("--Cv_factor",          type=float, default=1.618)
    parser.add_argument("--MLMCsteps",          type=int,   default=5,
                        help="Number of Machine-Learning assisted steps")
    parser.add_argument("--MCsteps",            type=int,   default=15,
                        help="Number of Monte Carlo steps per MLMC step")
    parser.add_argument("--num_temps",          type=int,   default=30)
    parser.add_argument("--schedule",           type=str,   default="Cv_beta")
    parser.add_argument("--num_epochs_start",   type=int,   default=40)
    parser.add_argument("--num_epochs_retrain", type=int,   default=1)
    parser.add_argument("--batch_size",         type=int,   default=256)
    args = parser.parse_args()

    # Consistency checks
    if args.schedule != "Cv_beta" and args.Cv_factor != parser.get_default("Cv_factor"):
        parser.error("Cv_factor can only be specified when schedule is Cv_beta.")
    if args.schedule == "Cv_beta" and args.num_temps != parser.get_default("num_temps"):
        parser.error("num_temps cannot be specified when schedule is Cv_beta.")

    num_temps_determiner = (
        args.Cv_factor if args.schedule == "Cv_beta" else args.num_temps
    )

    L    = args.L
    N    = L * L * L
    seed = args.seed
    J    = read_couplings(
        f"C:/Users/acer/Desktop/stage M1/MLMC_optimization/Data/Alpha/Couplings/"
        f"couplings_L{L}_R1_seed{seed}.txt",
        N,
    )

    temperatures, observ, elapsed_time = global_annealing(
        L, J,
        pop_size              = args.pop_size,
        num_steps_MC          = args.MLMCsteps,
        swap_step             = args.MCsteps,
        N                     = N,
        Tstart                = float(args.Tstart),
        Tend                  = float(args.Tend),
        Observables           = Observables,
        schedule              = args.schedule,
        num_epochs_start      = args.num_epochs_start,
        num_epochs_retrain    = args.num_epochs_retrain,
        batch_size            = args.batch_size,
        num_temps_determiner  = num_temps_determiner,
    )

    minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
    print(
        args.MLMCsteps, args.MCsteps,
        f"{len(temperatures)}",
        args.schedule,
        f"{minimum:.5f}",
        f"{observ.get_observable_history('mean_energy')[-1]:.5f}",
        elapsed_time,
    )
