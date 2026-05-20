import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import sys


sys.path.append("../../../Code/Legacy/packages")

from geometry import *
from utilities import *
from data_loads import *
from monte_carlo import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
def compute_energy_ST(s, J,h, J_trotter):
    """
    Vectorized Suzuki-Trotter energy for all replicas and all Trotter slices.

    Args:
        s         : [pop, P, N]  spin tensor (±1)
        J         : [N, N]       classical coupling matrix
        J_trotter : scalar       effective transverse coupling = -T/(2P) * ln(tanh(beta*Gamma/P))

    Returns:
        E_total : [pop]  total energy per replica (summed over Trotter slices)
    """
    # Classical Ising energy: E_cl[pop, P] = -0.5 * sum_ij J_ij s_i s_j
    # einsum "bpi, ij, bpj -> bp" but factored for efficiency:
    Js = torch.einsum("ij, bpj -> bpi", J, s)          # [pop, P, N]
    E_classical = -0.5 * (s * Js).sum(dim=-1)-torch.einsum("ki,i->k", s, h)       # [pop, P]

    # Trotter coupling: -J_T * sum_p s[p] * s[p+1]  (periodic in slice dim)
    s_next = torch.roll(s, -1, dims=1)                  # [pop, P, N]
    E_trotter = -J_trotter * (s * s_next).sum(dim=-1)   # [pop, P]

    # Sum over Trotter slices → [pop]
    return (E_classical + E_trotter).sum(dim=1)
'''


def compute_energy_ST(s, J, h, J_trotter):
    """
    Suzuki-Trotter energy for all replicas.

    Args:
        s         : [pop, P, N]
        J         : [N, N]
        h         : [N]
        J_trotter : scalar

    Returns:
        E_total : [pop]  energy per replica
    """
    # --- Classical J coupling: [pop, P] ---
    Js = torch.einsum("ij, bpj -> bpi", J, s)               # [pop, P, N]
    E_J = -0.5 * (s * Js).sum(dim=-1)                       # [pop, P]

    # --- Classical magnetic field: [pop, P] ---
    E_h = -(s * h).sum(dim=-1)                              # [pop, P]  h broadcasts over [pop,P,N]

    # --- Classical energy averaged over slices (not summed): [pop] ---
    E_classical = (E_J + E_h).mean(dim=1)                   # [pop]

    # --- Trotter coupling (quantum term): sum over slices IS correct here ---
    s_next = torch.roll(s, -1, dims=1)                      # [pop, P, N]
    E_trotter = -J_trotter * (s * s_next).sum(dim=(-1, -2)) # [pop]  sum over P and N

    return E_classical + E_trotter



def compute_delta_energy_ST(s, proposed, J,h, J_trotter, indices):
    """
    Vectorized ΔE for a checkerboard flip on `indices` across all replicas/slices.

    Args:
        s        : [pop, P, N]
        proposed : [pop, P, N]  s with indices flipped
        indices  : 1-D LongTensor of spin positions being updated

    Returns:
        delta_E : [pop, P, len(indices)]
    """
    # Classical contribution for flipped spins only
    # field[b,p,k] = sum_j J[k,j] * s[b,p,j]
    field = torch.einsum("kj, bpj -> bpk", J[indices], s)          # [pop, P, |idx|]
    # --- Classical: magnetic field ---
    # ΔE_h = -2 * h[k] * s_proposed[b,p,k]
    h_term = h[indices]                                               # [|idx|]
    dE_h   = 2.0 * proposed[:, :, indices] * h_term   
    dE_cl = -2.0 * (proposed[:, :, indices] * field+ dE_h)             # [pop, P, |idx|]

    # Trotter coupling: only the slice-neighbour terms for the flipped spins
    s_next = torch.roll(s, -1, dims=1)                              # [pop, P, N]
    s_prev = torch.roll(s,  1, dims=1)                              # [pop, P, N]
    dE_tr  = -J_trotter * proposed[:, :, indices] * (
        s_next[:, :, indices] + s_prev[:, :, indices]
    ) * 2                                                            # [pop, P, |idx|]
    # Factor of 2 because both neighbours change sign contribution
    return dE_cl + dE_tr  
                                         # [pop, P, |idx|]



# ─────────────────────────────────────────────
# Checkerboard MC update — no Python loop over spins
# ─────────────────────────────────────────────

def mc_update_ST(population, J,h, beta, J_trotter, INDICES):
    """
    One full checkerboard Metropolis sweep over the 3-D Suzuki-Trotter replica tensor.

    Args:
        population : [pop, P, N]
        J          : [N, N]
        beta       : scalar
        J_trotter  : scalar
        even/odd_indices : 1-D LongTensors

    Returns:
        updated population : [pop, P, N]
    """
    pop = population.detach().clone()
    for indices in INDICES:
        proposed = pop.detach().clone()
        proposed[:, :, indices] *= -1                               # [pop, P, |idx|]

        dE = compute_delta_energy_ST(pop, proposed, J,h, J_trotter, indices)  # [pop, P, |idx|]

        # Metropolis: accept each spin independently across pop, P, |idx|
        log_rand = torch.log(torch.rand_like(dE)).to(device)
        accept   = (dE < 0) | (log_rand < -beta * dE)              # [pop, P, |idx|]

        pop[:, :, indices] = torch.where(accept, proposed[:, :, indices], pop[:, :, indices])

    return pop


# ─────────────────────────────────────────────
# Effective Trotter coupling (analytic formula)
# ─────────────────────────────────────────────

def trotter_coupling(beta, Gamma, P):
    """
    J_T = -(T / 2P) * ln( tanh(beta * Gamma / P) )
        = (1 / 2*beta*P) * ln( coth(beta * Gamma / P) )   [equivalent]

    Returns scalar (float or 0-dim tensor).
    Safe-guarded against tanh → 0 at very large beta*Gamma/P.
    """
    x = beta * Gamma / P
    x = max(float(x), 1e-8)
    return -1.0 / (2.0 * beta) * np.log(np.tanh(x))


# ─────────────────────────────────────────────
# Main annealing loop
# ─────────────────────────────────────────────

def quantum_annealing_ST(
    K, J,h, pop_size, P,
    num_steps_MC, Tstart, Tend,
    Gamma_start, Gamma_end,
    Observables,
    schedule_T="linearT",
    schedule_Gamma="exp",
    num_temps=30,
    high_temp_therm=200,
    device=device,
):
    """
    Suzuki-Trotter simulated quantum annealing.

    Spin tensor shape: [pop_size, P, N]  (P = number of Trotter slices)
    All energy computations are fully vectorized — no Python loops over spins or slices.

    Args:
        L               : lattice side length
        J               : [N, N] coupling matrix (on device)
        pop_size        : number of independent replicas
        P               : number of Trotter slices (controls QA accuracy)
        num_steps_MC    : MC sweeps per temperature step
        Tstart / Tend   : temperature range
        Gamma_start/end : transverse-field schedule
        Observables     : class with .update(pop_2d) and .get_observable_history()
        schedule_T      : "linearT" | "linearBeta" | "Cv_beta"
        schedule_Gamma  : "linear" | "exp"
        num_temps       : number of annealing steps
        high_temp_therm : thermalization sweeps at Tstart
        dimension       : "2d" | "3d"
        device          : torch device string
    """
    J = J.to(device)

    N=K*2
    INDICES=greedy_independent_sets(J)[0]

    # ── Temperature schedule ──────────────────────────────────────────────
    # Reuse your existing schedule_temperatures helper
    temperatures = schedule_temperatures(Tstart, Tend, num_temps, schedule_T, N)

    # ── Gamma schedule ────────────────────────────────────────────────────
    n = len(temperatures)
    if schedule_Gamma == "linear":
        Gammas = torch.linspace(Gamma_start, Gamma_end, n)
    elif schedule_Gamma == "exp":
        Gammas = Gamma_start * (Gamma_end / Gamma_start) ** torch.linspace(0, 1, n)
    else:
        raise ValueError("Unknown Gamma schedule")

    # ── Initialize 3-D spin tensor [pop, P, N] ───────────────────────────
    population = (torch.randint(0, 2, (pop_size, P, N), device=device).float() * 2 - 1)

    observ = Observables(J, N,h)

    # ── High-temperature thermalization ───────────────────────────────────
    beta0   = 1.0 / temperatures[0]
    Gamma0  = float(Gammas[0])
    J_T0    = trotter_coupling(beta0, Gamma0, P)
    for _ in range(high_temp_therm):
        population = mc_update_ST(population, J,h, beta0, J_T0, INDICES)

    # Record initial state (collapse slices by taking mean config or best-energy slice)
    best_slice = _best_slice(population, J,h)
    observ.update(best_slice)

    # ── Annealing loop ────────────────────────────────────────────────────
    start_time = time.time()

    for t, (T, Gamma) in enumerate(zip(temperatures, Gammas)):
        beta  = 1.0 / float(T)
        Gamma = float(Gamma)
        J_T   = trotter_coupling(beta, Gamma, P)

        for _ in range(num_steps_MC):
            population = mc_update_ST(population, J,h, beta, J_T, INDICES)

        best_slice = _best_slice(population, J,h)
        observ.update(best_slice)
    elapsed = time.time() - start_time

    # ── Extract best configuration ─────────────────────────────────────────
    # population: [pop, P, N] → pick slice with lowest classical energy per replica
    best_configs = _best_slice(population, J,h)   # [pop, N]
    return temperatures, observ, elapsed, best_configs


# ─────────────────────────────────────────────
# Helper: pick the Trotter slice with minimum classical energy
# ─────────────────────────────────────────────

def _best_slice(population, J,h):
    """
    For each replica, return the Trotter slice with the lowest classical Ising energy.

    Args:
        population : [pop, P, N]
        J          : [N, N]

    Returns:
        best : [pop, N]
    """
    pop_size, P, N = population.shape
    # Classical energy per (replica, slice): [pop, P]
    Js  = torch.einsum("ij, bpj -> bpi", J, population)    # [pop, P, N]
    # Magnetic field energy: [pop, P]
    E_h   = -(population * h).sum(dim=-1)  
    E_J   = -0.5 * (population * Js).sum(dim=-1)  
    E=E_J+E_h          # [pop, P]
    idx = E.argmin(dim=1)                                    # [pop]
    # Gather best slice
    best = population[torch.arange(pop_size), idx, :]       # [pop, N]
    return best


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suzuki-Trotter Quantum Annealing")
    parser.add_argument("--K",            type=int,   default=999)
    parser.add_argument("--seed",         type=int,   default=345692)
    parser.add_argument("--Tstart",       type=float, default=1.2)
    parser.add_argument("--Tend",         type=float, default=0.1)
    parser.add_argument("--MCsteps",      type=int,   default=500)
    parser.add_argument("--num_temps",    type=int,   default=50)
    parser.add_argument("--schedule",     type=str,   default="linearBeta")
    parser.add_argument("--pop_size",     type=int,   default=10000//40)
    parser.add_argument("--P",            type=int,   default=40,
                        help="Number of Trotter slices")
    parser.add_argument("--Gamma_start",  type=float, default=1.2)
    parser.add_argument("--Gamma_end",    type=float, default=0.01)
    parser.add_argument("--schedule_Gamma", type=str, default="exp")
    parser.add_argument("--device",       type=str,   default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    K = args.K
    N = K*2
    seed = args.seed
    J = read_couplings(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Data/Alpha/Couplings/couplings_XORSAT_K{K}_seed{seed}.txt', N).to(device)
    h=read_field(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Data/Alpha/Couplings/field_XORSAT_K{K}_seed{seed}.txt', N).to(device)
    


    temperatures, observ, elapsed_time, best_configs = quantum_annealing_ST(
        K=K, J=J,h=h,
        pop_size=args.pop_size,
        P=args.P,
        num_steps_MC=args.MCsteps,
        Tstart=args.Tstart,
        Tend=args.Tend,
        Gamma_start=args.Gamma_start,
        Gamma_end=args.Gamma_end,
        Observables=Observables,
        schedule_T=args.schedule,
        schedule_Gamma=args.schedule_Gamma,
        num_temps=args.num_temps,
        device=args.device)

    minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
    real_minimum = torch.tensor(observ.get_observable_history("min_real_energy")).min()
    print( f"{minimum:.5f}", f"{observ.get_observable_history('mean_energy')[-1]:.5f}", elapsed_time)
    print( f"{real_minimum:.5f}", f"{observ.get_observable_history('mean_real_energy')[-1]:.5f}", elapsed_time)
 
    print(
        args.MCsteps,
        args.P,
        len(temperatures),
        args.schedule,
        f"{minimum:.5f}",
        f"{observ.get_observable_history('mean_energy')[-1]:.5f}",
        elapsed_time,
    )