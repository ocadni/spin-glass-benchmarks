import torch
import numpy as np
import sys
import time

sys.path.append("../../../Code/Legacy/packages")
from utilities import *
from data_loads import *
from monte_carlo import *


def _reluctant_step(population, J):
    """
    Performs one reluctant greedy step vectorized over the population:
    Finds the spin with the smallest energy drop (Delta E < 0 closest to 0) and flips it.
    Completely independent of topology.
    """
    # Compute local fields: h_i = sum_j J_ij * s_j
    if J.is_sparse:
        local_fields = torch.sparse.mm(J, population.t()).t()
    else:
        local_fields = torch.matmul(population, J.t())

    # Delta E for flipping spin i: Delta E_i = 2 * s_i * h_i
    delta_E = 2.0 * population * local_fields

    # Add tiny random jitter to break ties randomly instead of index 0 bias
    jitter = torch.rand_like(delta_E) * 1e-9

    # Mask non-improving moves (Delta E >= 0) with -inf
    masked_delta_E = torch.where(
        delta_E < -1e-6,
        delta_E + jitter,
        torch.tensor(-float('inf'), device=delta_E.device)
    )

    # Pick spin with largest negative Delta E (smallest energy decrease)
    best_spins = torch.argmax(masked_delta_E, dim=1)

    # Check which configurations still have at least one valid downhill move
    improving_vals = masked_delta_E.gather(1, best_spins.unsqueeze(1)).squeeze(1)
    active_mask = improving_vals > -float('inf')

    if not active_mask.any():
        return population, False  # All replicas reached a local minimum

    # Flip the chosen spins
    row_idx = torch.arange(population.shape[0], device=population.device)[active_mask]
    col_idx = best_spins[active_mask]
    population[row_idx, col_idx] *= -1

    return population, True


def _random_step(population, J):
    """
    Performs one random sequential greedy step vectorized over the population:
    Selects a spin uniformly at random among all downhill moves (Delta E < 0) and flips it.
    Completely independent of topology.
    """
    # Compute local fields: h_i = sum_j J_ij * s_j
    if J.is_sparse:
        local_fields = torch.sparse.mm(J, population.t()).t()
    else:
        local_fields = torch.matmul(population, J.t())

    # Delta E for flipping spin i: Delta E_i = 2 * s_i * h_i
    delta_E = 2.0 * population * local_fields

    # Assign uniform random scores to improving moves, -inf to non-improving
    random_scores = torch.where(
        delta_E < -1e-6,
        torch.rand_like(delta_E),
        torch.tensor(-float('inf'), device=delta_E.device)
    )

    # Pick a random improving spin (the one with the largest random score)
    chosen_spins = torch.argmax(random_scores, dim=1)

    # Check which configurations still have at least one downhill move
    improving_vals = random_scores.gather(1, chosen_spins.unsqueeze(1)).squeeze(1)
    active_mask = improving_vals > -float('inf')

    if not active_mask.any():
        return population, False  # All replicas reached a local minimum

    # Flip the chosen spins
    row_idx = torch.arange(population.shape[0], device=population.device)[active_mask]
    col_idx = chosen_spins[active_mask]
    population[row_idx, col_idx] *= -1

    return population, True


def greedy_search(L, J, pop_size, num_sweeps, Observables, 
                  mode="random", record_interval=1):
    """
    Sequential greedy optimization algorithm (topology-independent).
    
    Parameters:
    -----------
    L : int or None
        Lattice length (kept for backwards compatibility, not required for topology).
    J : torch.Tensor
        Coupling matrix (N x N), dense or sparse.
    pop_size : int
        Number of parallel replicas.
    num_sweeps : int
        Number of sweeps (total single-spin updates = num_steps * N).
    Observables : class
        Observables tracker.
    mode : str
        - "random": Uniformly chooses a random downhill move (Delta E < 0).
        - "reluctant": Chooses the downhill move with the smallest energy decrease.
    record_interval : int
        Frequency of single-spin steps at which observables are recorded.
        Set to 1 to record every flip, or N to record once per sweep.
    """
    if mode not in ["random", "reluctant"]:
        raise ValueError("mode must be either 'random' or 'reluctant'")

    # Derive system size N directly from J (independent of dimensions/bipartiteness)
    N = J.shape[0]
    device = J.device

    # Initialize population directly on the same device as J
    population = torch.randint(0, 2, (pop_size, N), device=device).float() * 2 - 1
    
    # Initialize observables
    observ = Observables(J, N)

    # Total single-spin flips corresponding to `num_steps` full sweeps
    total_steps = num_sweeps * N

    # Select step function
    step_fn = _random_step if mode == "random" else _reluctant_step

    start_time = time.time()
    observ.set_start_time()

    # Sequential Optimization Loop
    for i in range(total_steps):
        population, changed = step_fn(population, J)
        
        # Early stop if all replicas are trapped in local minima
        if not changed:
            print(f"[{mode.upper()}] All replicas reached a local minimum at step {i} / {total_steps}.")
            observ.update(population)
            break

        if (i + 1) % record_interval == 0:
            observ.update(population)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return observ, elapsed_time