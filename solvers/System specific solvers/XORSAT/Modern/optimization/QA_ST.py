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


# =============================================================================
# SUZUKI-TROTTER QUANTUM ANNEALING
# =============================================================================
#
# THEORY RECAP:
# -------------
# The d-dimensional quantum Ising model is mapped to a (d+1)-dimensional
# classical Ising model via the Suzuki-Trotter decomposition.
#
# Quantum Hamiltonian:
#   H(s) = -A(s) * sum_i sigma_i^x  -  B(s) * sum_ij J_ij sigma_i^z sigma_j^z
#
# After the mapping, the effective classical Hamiltonian acting on P replicas is:
#   H_eff = -sum_{k=1}^{P} [ sum_ij J_ij sigma_i^(k) sigma_j^(k)
#                           + J_perp * sum_i sigma_i^(k) sigma_i^(k+1) ]
#
# where the inter-replica (Trotter) coupling is:
#   J_perp(Gamma, beta, P) = -(P / (2*beta)) * ln( tanh(beta * Gamma / P) )
#
# 
# The annealing schedule reduces Gamma from Gamma_start -> 0
#
# The population shape is (pop_size, P, N):
#   - pop_size : number of independent annealing runs  [same as SA]
#   - P        : number of Trotter replicas            [new dimension vs SA]
#   - N        : number of spins                       [same as SA]
#
# =============================================================================


def compute_J_perp(Gamma: float, beta: float, P: int) -> float:
    """
    Compute the effective inter-replica (Trotter) coupling J_perp.

    This coupling encodes the transverse field Gamma in the classical
    extended system. It plays the same structural role as the temperature
    in SA: large Gamma -> large J_perp -> replicas disagree easily
    -> tunneling is cheap.

    Args:
        Gamma : transverse field strength (annealing control parameter)
        beta  : fictitious inverse temperature (fixed, NOT the schedule var)
        P     : number of Trotter replicas

    Returns:
        J_perp : scalar float
    """
    arg = np.tanh(beta * Gamma / P)
    # Guard against log(0) when Gamma -> 0
    arg = max(arg, 1e-15)
    return -(P / (2.0 * beta)) * np.log(arg)


def schedule_transverse_field(
    Gamma_start: float,
    Gamma_end: float,
    num_steps: int,
    schedule: str = "linear"
) -> np.ndarray:
    """
    Build the Gamma (transverse field) annealing schedule.
      - "linear"      : uniform steps in Gamma
      - "exponential" : geometric decay (analogous to log schedule in beta)

    Args:
        Gamma_start : initial (large) transverse field
        Gamma_end   : final (near-zero) transverse field
        num_steps   : number of annealing steps
        schedule    : "linear" or "exponential"

    Returns:
        gammas : 1D array of length num_steps
    """
    if schedule == "linear":
        gammas = np.linspace(Gamma_start, Gamma_end, num_steps)
    elif schedule == "exponential":
        gammas = np.exp(np.linspace(np.log(Gamma_start), np.log(Gamma_end), num_steps))
    else:
        raise ValueError("schedule must be 'linear' or 'exponential'")
    return gammas


def trotter_interreplica_update(
    population: torch.Tensor,
    J_perp: float,
    beta: float,
    P: int
) -> torch.Tensor:
    """
    Metropolis sweep along the imaginary-time (Trotter) axis.

    For each spin site i, we propose flipping sigma_i^(k) and accept/reject
    based on the energy change from inter-replica bonds only:
      delta_E_perp = 2 * J_perp * sigma_i^(k) * (sigma_i^(k-1) + sigma_i^(k+1))

    This is the purely quantum (tunneling) part of the update, separate from
    the intra-slice spatial update that handles the problem Hamiltonian J.

    Periodic boundary conditions are used in the Trotter direction
    (the imaginary-time axis is a ring of P slices).

    Args:
        population : (pop_size, P, N) tensor of ±1 spins
        J_perp     : inter-replica coupling (encodes current Gamma)
        beta       : fictitious inverse temperature
        P          : number of Trotter replicas

    Returns:
        population : updated (pop_size, P, N) tensor
    """
    pop_size, P_, N = population.shape

    # Neighbour replicas with periodic BC in imaginary time
    # prev_replica[k] = k-1 mod P,  next_replica[k] = k+1 mod P
    prev_replica = population[:, (torch.arange(P_) - 1) % P_, :]  # (pop_size, P, N)
    next_replica = population[:, (torch.arange(P_) + 1) % P_, :]  # (pop_size, P, N)

    # Energy cost of flipping sigma_i^(k):
    # delta_E = 2 * J_perp * sigma_i^(k) * (sigma_i^(k-1) + sigma_i^(k+1))
    delta_E = 2.0 * J_perp * population * (prev_replica + next_replica)  # (pop_size, P, N)

    # Metropolis acceptance
    accept_prob = torch.exp(-beta * delta_E).clamp(max=1.0)
    flip_mask = torch.rand_like(accept_prob) < accept_prob

    population = torch.where(flip_mask, -population, population)
    return population


def quantum_annealing_suzuki_trotter(
    L: int,
    J: torch.Tensor,
    pop_size: int,
    num_steps_MC: int,
    N: int,
    Gamma_start: float,
    Gamma_end: float,
    beta: float,
    P: int,
    Observables,
    schedule: str = "linear",
    high_temp_thermalization_steps: int = 200,
    dimension: str = "3d"
):
    """
    Suzuki-Trotter Quantum Annealing.

    Mirrors simulated_annealing() exactly in structure:
      - same checkerboard spatial update via monte_carlo_update_fast()
      - same Observables bookkeeping
      - same thermalization block at high Gamma
      - same loop structure: for each Gamma, do num_steps_MC spatial updates

    The key differences from SA:
      1. population shape is (pop_size, P, N) instead of (pop_size, N).
         The P dimension is the Trotter (imaginary-time) axis.
      2. Each MC step consists of TWO sub-updates:
           a. Spatial update  (intra-slice, handles J)  — same as SA
           b. Trotter update  (inter-replica, handles Gamma via J_perp)
      3. The annealing parameter is Gamma (transverse field), not T.
         beta is fixed throughout.
      4. Observables are evaluated on the replica-averaged spin configuration
         (or optionally on all replicas independently).

    Args:
        L            : linear lattice size
        J            : coupling matrix, same format as SA
        pop_size     : number of independent runs
        num_steps_MC : MC sweeps per annealing step (same role as SA)
        N            : number of spins (L^3 or L^2)
        Gamma_start  : initial transverse field (large -> quantum regime)
        Gamma_end    : final transverse field (near 0 -> classical regime)
        beta         : fictitious inverse temperature (FIXED, not annealed)
        P            : number of Trotter replicas
        Observables  : same Observables class used in SA
        schedule     : "linear" or "exponential" Gamma schedule
        high_temp_thermalization_steps : thermalization sweeps at Gamma_start
        dimension    : "3d" or "2d"

    Returns:
        gammas      : array of Gamma values used
        observ      : filled Observables object
        elapsed_time: wall-clock time in seconds
    """

    # -------------------------------------------------------------------------
    # 1. Geometry — identical to SA
    # -------------------------------------------------------------------------
    if dimension == "3d":
        even_indices, odd_indices = get_indices(L)
        N = L * L * L
    elif dimension == "2d":
        even_indices, odd_indices = get_indices_2D(L)
        N = L * L
    else:
        raise ValueError("dimension must be either 3d or 2d")

    # -------------------------------------------------------------------------
    # 2. Gamma schedule — analogous to schedule_temperatures() in SA
    # -------------------------------------------------------------------------
    gammas = schedule_transverse_field(Gamma_start, Gamma_end, num_steps_MC, schedule)

    # -------------------------------------------------------------------------
    # 3. Population — shape (pop_size, P, N)
    #    The P axis is NEW compared to SA's (pop_size, N).
    #    Each of the P replicas is an independent copy of the spin system
    #    that is coupled to its neighbors along the Trotter axis.
    # -------------------------------------------------------------------------
    population = torch.randint(0, 2, (pop_size, P, N)).float() * 2 - 1

    # -------------------------------------------------------------------------
    # 4. Observables — same as SA, but we collapse replica dimension first
    # -------------------------------------------------------------------------
    observ = Observables(J, N)

    # -------------------------------------------------------------------------
    # 5. Thermalization at Gamma_start — same role as SA's high-T thermalization
    #    At large Gamma, J_perp is large and replicas fluctuate freely.
    # -------------------------------------------------------------------------
    J_perp_init = compute_J_perp(Gamma_start, beta, P)

    for _ in range(high_temp_thermalization_steps):
        # 5a. Spatial update for each replica slice independently
        for k in range(P):
            population[:, k, :] = monte_carlo_update_fast(
                population[:, k, :], J,
                beta=beta,
                even_indices=even_indices,
                odd_indices=odd_indices
            )
        # 5b. Trotter (inter-replica) update
        population = trotter_interreplica_update(population, J_perp_init, beta, P)

    # Evaluate observables on the mean-field replica (slice 0 as representative)
    observ.update(population[:, 0, :])

    # -------------------------------------------------------------------------
    # 6. Main annealing loop — mirrors SA's `for T in temperatures[1:]:`
    #    Gamma decreases -> J_perp decreases -> replicas align -> tunneling off
    # -------------------------------------------------------------------------
    start_time = time.time()

    for Gamma in gammas[1:]:
        # Recompute inter-replica coupling for current Gamma
        J_perp = compute_J_perp(Gamma, beta, P)

        for _ in range(num_steps_MC):
            # 6a. Spatial update (intra-slice) — handles problem Hamiltonian J
            #     Identical to SA's monte_carlo_update_fast call
            for k in range(P):
                population[:, k, :] = monte_carlo_update_fast(
                    population[:, k, :], J,
                    beta=beta,
                    even_indices=even_indices,
                    odd_indices=odd_indices
                )
            # 6b. Trotter update (inter-replica) — handles transverse field Gamma
            #     This has NO counterpart in SA; it is the quantum tunneling step
            population = trotter_interreplica_update(population, J_perp, beta, P)

        # Track observables on replica 0 (equivalent to a single classical config)
        observ.update(population[:, 0, :])

    end_time = time.time()
    elapsed_time = end_time - start_time

    return gammas, observ, elapsed_time


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suzuki-Trotter Quantum Annealing")

    # --- Arguments shared with SA ---
    parser.add_argument("--pop_size",  type=int,   default=1000,      help="Population size")
    parser.add_argument("--L",         type=int,   default=10,         help="Lattice size")
    parser.add_argument("--seed",      type=int,   default=310411727,  help="Random seed")
    parser.add_argument("--MCsteps",   type=int,   default=320,        help="MC sweeps per Gamma step")
    parser.add_argument("--num_temps", type=int,   default=30,         help="Number of annealing Gamma values")
    parser.add_argument("--schedule",  type=str,   default="linear",   help="Gamma schedule: linear or exponential")

    # --- Arguments specific to QA (replacing T with Gamma) ---
    parser.add_argument("--Gamma_start", type=float, default=2.0,  help="Starting transverse field (large = quantum)")
    parser.add_argument("--Gamma_end",   type=float, default=0.01, help="Ending transverse field (small = classical)")
    parser.add_argument("--beta",        type=float, default=1.0,  help="Fictitious inverse temperature (FIXED)")
    parser.add_argument("--P",           type=int,   default=20,   help="Number of Trotter replicas")

    args = parser.parse_args()

    # --- Setup (same as SA) ---
    L    = args.L
    N    = L * L * L
    seed = args.seed
    J    = read_couplings(
        f"C:/Users/acer/Desktop/stage M1/MLMC_optimization/Data/Alpha/Couplings/"
        f"couplings_L{L}_R1_seed{seed}.txt", N
    )

    # --- Run QA ---
    gammas, observ, elapsed_time = quantum_annealing_suzuki_trotter(
        L            = L,
        J            = J,
        pop_size     = args.pop_size,
        num_steps_MC = args.MCsteps,
        N            = N,
        Gamma_start  = args.Gamma_start,
        Gamma_end    = args.Gamma_end,
        beta         = args.beta,
        P            = args.P,
        Observables  = Observables,
        schedule     = args.schedule,
    )
#         num_temps_determiner = args.num_temps,
    minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
    print(
        args.MCsteps,
        f"{len(gammas)}",
        args.schedule,
        f"{minimum:.5f}",
        f'{observ.get_observable_history("mean_energy")[-1]:.5f}',
        elapsed_time
    )