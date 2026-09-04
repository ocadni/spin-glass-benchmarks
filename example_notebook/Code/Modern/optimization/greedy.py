import torch
import numpy as np
import sys
import time
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve()
_NOTEBOOK_ROOT = _HERE.parents[3]
_REPO_ROOT = _HERE.parents[4]
_LEGACY_PACKAGES = _NOTEBOOK_ROOT / "Code" / "Legacy" / "packages"

for _path in (_LEGACY_PACKAGES, _REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from utilities import *
from monte_carlo import Observables, read_couplings
from device_utils import get_best_device, get_device, synchronize


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

    synchronize(device)
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

    synchronize(device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return observ, elapsed_time


def _load_canonical_instance(path, device, symmetric=True):
    from generators.pairwise_io import load_pairwise_instance

    instance = load_pairwise_instance(path)
    couplings = torch.zeros(instance.num_spins, instance.num_spins, device=device)
    for interaction in instance.interactions:
        couplings[interaction.i, interaction.j] = interaction.coupling
        if symmetric:
            couplings[interaction.j, interaction.i] = interaction.coupling

    if any(field != 0 for field in instance.fields):
        print("warning: nonzero fields are present, but this greedy implementation uses only pairwise couplings")

    return couplings, instance.num_spins, "canonical"


def _infer_num_spins(args):
    if args.num_spins is not None:
        return args.num_spins
    if args.lattice_size is None:
        return None
    if args.dimension == "2d":
        return args.lattice_size * args.lattice_size
    if args.dimension == "3d":
        return args.lattice_size * args.lattice_size * args.lattice_size
    raise ValueError("dimension must be either 2d or 3d")


def load_coupling_matrix(path, device, input_format="auto", num_spins=None, start_from_one=False, symmetric=True):
    path = Path(path)
    if input_format in {"auto", "canonical"}:
        try:
            return _load_canonical_instance(path, device=device, symmetric=symmetric)
        except Exception:
            if input_format == "canonical":
                raise

    if num_spins is None:
        raise ValueError(
            "num_spins is required for plain edge-list files; pass --num-spins or --lattice-size"
        )

    couplings = read_couplings(path, num_spins, start_from_one=start_from_one).to(device)
    return couplings, num_spins, "edge-list"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the notebook greedy algorithm on a pairwise spin-glass instance."
    )
    parser.add_argument("instance", type=Path, help="Path to an instance file.")
    parser.add_argument("--pop-size", type=int, default=1000, help="Number of parallel replicas.")
    parser.add_argument("--sweeps", type=int, default=5, help="Number of greedy sweeps.")
    parser.add_argument("--mode", choices=["random", "reluctant"], default="random")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device. auto prefers MPS, then CUDA, then CPU.",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "canonical", "edge-list"],
        default="auto",
        help="canonical reads benchmark files; edge-list reads old three-column files.",
    )
    parser.add_argument(
        "--num-spins",
        type=int,
        default=None,
        help="Number of spins for old plain edge-list files.",
    )
    parser.add_argument(
        "--lattice-size",
        type=int,
        default=None,
        help="Side length L for old edge-list files when --num-spins is omitted.",
    )
    parser.add_argument(
        "--dimension",
        choices=["2d", "3d"],
        default="3d",
        help="Dimension used with --lattice-size for old edge-list files.",
    )
    parser.add_argument(
        "--start-from-one",
        action="store_true",
        help="Treat old edge-list indices as one-based.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for PyTorch and NumPy.")
    parser.add_argument(
        "--record-interval",
        type=int,
        default=None,
        help="Record observables every this many single-spin steps. Default: once per sweep.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = get_best_device() if args.device == "auto" else get_device(args.device)
    num_spins = _infer_num_spins(args)
    couplings, num_spins, input_format = load_coupling_matrix(
        args.instance,
        device=device,
        input_format=args.format,
        num_spins=num_spins,
        start_from_one=args.start_from_one,
        symmetric=True,
    )
    record_interval = args.record_interval if args.record_interval is not None else num_spins

    observ, elapsed_time = greedy_search(
        L=args.lattice_size,
        J=couplings,
        pop_size=args.pop_size,
        num_sweeps=args.sweeps,
        Observables=Observables,
        mode=args.mode,
        record_interval=record_interval,
    )

    min_history = observ.get_observable_history("min_energy")
    mean_history = observ.get_observable_history("mean_energy")
    print(f"instance: {args.instance}")
    print(f"format: {input_format}")
    print(f"device: {couplings.device}")
    print(f"N: {num_spins}")
    print(f"mode: {args.mode}")
    print(f"pop_size: {args.pop_size}")
    print(f"sweeps: {args.sweeps}")
    print(f"records: {len(min_history)}")
    print(f"min_energy_per_spin: {min(min_history):.8f}")
    print(f"final_mean_energy_per_spin: {mean_history[-1]:.8f}")
    print(f"elapsed_seconds: {elapsed_time:.6f}")


if __name__ == "__main__":
    main()
