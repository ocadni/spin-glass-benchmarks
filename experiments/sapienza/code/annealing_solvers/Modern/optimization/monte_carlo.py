#Code to perform standard Metropolis MC, with optimization for parallelization
#In particular, we use a checkerboard update, where we update even and odd indices separately

import torch
import numpy as np
import sys
import time
from collections import deque
from functools import partial

sys.path.append("../../../Code/Legacy/packages")
from utilities import compute_energy as _compute_energy_zero_field

def get_num_spins(J):
    """Infer the system size from a square coupling matrix."""
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J must be a square coupling matrix")
    return J.shape[0]


def normalize_external_fields(J, fields=None):
    """Validate fields and return ``None`` for the zero-field case.

    ``None`` is the representation used throughout the annealers for an
    instance with no effective external field.  This keeps the original
    zero-field implementations on their existing code path.
    """
    if fields is None:
        return None
    if fields.ndim != 1 or fields.numel() != get_num_spins(J):
        raise ValueError("fields must be a vector with one entry per spin")
    fields = fields.to(device=J.device, dtype=J.dtype)
    return fields if bool(torch.count_nonzero(fields).item()) else None


def compute_energy_zero_field(population, J, take_mean=False):
    """Compute the original pairwise-only Ising energy."""
    return _compute_energy_zero_field(population, J, take_mean=take_mean)


def compute_energy_with_fields(population, J, fields, take_mean=False):
    """Compute ``-1/2 s^T J s - h^T s`` for each population member."""
    fields = normalize_external_fields(J, fields)
    if fields is None:
        return compute_energy_zero_field(population, J, take_mean=take_mean)
    energy = compute_energy_zero_field(population, J, take_mean=False)
    energy = energy - torch.einsum("ki,i->k", population, fields)
    return energy.mean() if take_mean else energy


def select_energy_function(J, fields=None):
    """Return the zero-field or field-aware energy function for an instance."""
    fields = normalize_external_fields(J, fields)
    if fields is None:
        return compute_energy_zero_field, None
    return partial(compute_energy_with_fields, fields=fields), fields


def check_bipartite(J, atol=0.0, return_coloring=False):
    """Check whether the nonzero off-diagonal couplings in ``J`` are bipartite.

    The coupling matrix is interpreted as an undirected graph: an edge connects
    spins ``i`` and ``j`` when either ``J[i, j]`` or ``J[j, i]`` is nonzero
    (up to ``atol``).  A breadth-first two-colouring detects an edge whose
    endpoints would need the same colour, in which case the graph is not
    bipartite.  Diagonal entries are ignored because they are constant for
    Ising spins.

    With ``return_coloring=True``, return ``(is_bipartite, colors)``.  Colors
    is a CPU tensor containing 0 or 1 for each spin when successful, and None
    otherwise.
    """
    get_num_spins(J)
    if atol < 0:
        raise ValueError("atol must be non-negative")

    # Copy the adjacency relation to CPU once; this avoids a device
    # synchronization for every vertex during the graph traversal.
    adjacency = (J.detach().abs() > atol).to(device="cpu")
    adjacency = torch.logical_or(adjacency, adjacency.T)
    adjacency.fill_diagonal_(False)

    num_spins = J.shape[0]
    colors = torch.full((num_spins,), -1, dtype=torch.int8)
    for start in range(num_spins):
        if colors[start] != -1:
            continue

        colors[start] = 0
        queue = deque([start])
        while queue:
            spin = queue.popleft()
            neighbor_indices = torch.nonzero(adjacency[spin], as_tuple=False).flatten().tolist()
            for neighbor in neighbor_indices:
                if colors[neighbor] == -1:
                    colors[neighbor] = 1 - colors[spin]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[spin]:
                    if return_coloring:
                        return False, None
                    return False

    if return_coloring:
        return True, colors
    return True


def get_bipartite_spin_groups(J, atol=0.0):
    """Return two independent spin groups, or ``(None, None)`` if impossible."""
    is_bipartite, colors = check_bipartite(J, atol=atol, return_coloring=True)
    if not is_bipartite:
        return None, None

    even_indices = torch.nonzero(colors == 0, as_tuple=False).flatten().to(J.device)
    odd_indices = torch.nonzero(colors == 1, as_tuple=False).flatten().to(J.device)
    return even_indices, odd_indices


def monte_carlo_update_fast_zero_field(pop, J, beta, even_indices, odd_indices):
    """Perform the original zero-field checkerboard Metropolis sweep."""
    population = pop.clone()
    pop_size, N = population.shape
    # Define "even" and "odd" indices for a checkerboard update

    # Update spins in two passes (checkerboard pattern)
    for indices in [even_indices, odd_indices]:
        # Propose flips for the entire population at selected indices
        proposed_population = population.clone()
        proposed_population[:, indices] *= -1
        
        # Compute energy difference for each single-spin flip
        delta_E = -2* torch.einsum("ki, ki->ki",proposed_population[:, indices], torch.einsum("kj, ji->ki", population, J[indices, :].T))
        
        # Metropolis acceptance criterion for each spin
        acceptance_prob = torch.exp(-beta * delta_E)
        random_vals = torch.rand(pop_size, len(indices), device=population.device)
        accept = (delta_E < 0) | (random_vals < acceptance_prob)
        
        # Apply accepted flips only for accepted positions
        population[:, indices] = torch.where(accept, proposed_population[:, indices], population[:, indices])

    return population


def monte_carlo_update_fast_with_fields(pop, J, beta, even_indices, odd_indices, fields):
    """Perform a checkerboard Metropolis sweep including ``-h^T s``."""
    population = pop.clone()
    pop_size, _ = population.shape

    for indices in [even_indices, odd_indices]:
        proposed_population = population.clone()
        proposed_population[:, indices] *= -1
        delta_E = -2 * torch.einsum(
            "ki, ki->ki",
            proposed_population[:, indices],
            torch.einsum("kj, ji->ki", population, J[indices, :].T),
        )
        delta_E = delta_E + 2 * population[:, indices] * fields[indices]
        acceptance_prob = torch.exp(-beta * delta_E)
        random_vals = torch.rand(pop_size, len(indices), device=population.device)
        accept = (delta_E < 0) | (random_vals < acceptance_prob)
        population[:, indices] = torch.where(
            accept, proposed_population[:, indices], population[:, indices]
        )

    return population


def monte_carlo_update_random_zero_field(pop, J, beta, even_indices=None, odd_indices=None):
    """Perform one random-order, sequential single-spin Metropolis sweep.

    Each spin is considered once in a fresh random order.  Updates remain
    sequential, so this is valid even if every pair of spins is coupled.  The
    optional group arguments give this function the same call signature as the
    checkerboard updater and are intentionally ignored.
    """
    del even_indices, odd_indices
    population = pop.clone()
    pop_size, num_spins = population.shape

    for spin in torch.randperm(num_spins, device=population.device):
        # This is the energy difference E(-s_i) - E(s_i) for the convention
        # used by monte_carlo_update_fast.
        local_field = torch.matmul(population, J[spin, :])
        delta_E = 2 * population[:, spin] * local_field
        acceptance_prob = torch.exp(-beta * delta_E)
        accept = (delta_E < 0) | (
            torch.rand(pop_size, device=population.device) < acceptance_prob
        )
        population[:, spin] = torch.where(accept, -population[:, spin], population[:, spin])

    return population


def monte_carlo_update_random_with_fields(pop, J, beta, even_indices=None,
                                          odd_indices=None, fields=None):
    """Perform a random-order sequential Metropolis sweep with fields."""
    del even_indices, odd_indices
    fields = normalize_external_fields(J, fields)
    if fields is None:
        return monte_carlo_update_random_zero_field(pop, J, beta)

    population = pop.clone()
    pop_size, num_spins = population.shape
    for spin in torch.randperm(num_spins, device=population.device):
        local_field = torch.matmul(population, J[spin, :]) + fields[spin]
        delta_E = 2 * population[:, spin] * local_field
        acceptance_prob = torch.exp(-beta * delta_E)
        accept = (delta_E < 0) | (
            torch.rand(pop_size, device=population.device) < acceptance_prob
        )
        population[:, spin] = torch.where(accept, -population[:, spin], population[:, spin])

    return population


# Backward-compatible names for direct zero-field callers.
monte_carlo_update_fast = monte_carlo_update_fast_zero_field
monte_carlo_update_random = monte_carlo_update_random_zero_field


def _select_monte_carlo_update(J, fast_update, random_update, atol=0.0):
    """Select a graph-valid local updater from a zero- or field-aware pair."""
    even_indices, odd_indices = get_bipartite_spin_groups(J, atol=atol)
    if even_indices is None:
        return random_update, None, None
    return fast_update, even_indices, odd_indices


def select_monte_carlo_update_zero_field(J, atol=0.0):
    """Select the original zero-field local Metropolis implementation."""
    return _select_monte_carlo_update(
        J, monte_carlo_update_fast_zero_field, monte_carlo_update_random_zero_field, atol
    )


def select_monte_carlo_update_with_fields(J, fields, atol=0.0):
    """Select a field-aware local Metropolis implementation."""
    fields = normalize_external_fields(J, fields)
    if fields is None:
        return select_monte_carlo_update_zero_field(J, atol=atol)
    return _select_monte_carlo_update(
        J,
        partial(monte_carlo_update_fast_with_fields, fields=fields),
        partial(monte_carlo_update_random_with_fields, fields=fields),
        atol,
    )


def select_monte_carlo_update(J, fields=None, atol=0.0):
    """Select zero-field or field-aware local MC and its spin groups.

    Returns ``(update_function, first_group, second_group)``.  Bipartite
    coupling graphs use the existing parallel checkerboard updater; all other
    graphs use random-order sequential single-spin updates and return None for
    both groups.
    """
    fields = normalize_external_fields(J, fields)
    if fields is None:
        return select_monte_carlo_update_zero_field(J, atol=atol)
    return select_monte_carlo_update_with_fields(J, fields, atol=atol)

def read_couplings(file, *, start_from_one=False, return_fields=False,
                   device=None, dtype=torch.float32):
    """Load a benchmark instance into a dense coupling matrix.

    Benchmark files begin with a ``#`` metadata header containing ``N`` and
    ``num_fields``.  The next ``num_fields`` rows are ``spin_index h_i`` field
    entries; all remaining rows are ``spin_i spin_j J_ij`` couplings.  ``N`` is
    therefore inferred from the file, never supplied by the caller.

    By default, return the symmetric coupling matrix ``J``.  Set
    ``return_fields=True`` to return ``(J, h)``.  Here ``h`` is ``None`` when
    every field is zero, and otherwise is the external-field vector.  This
    lets callers select the original zero-field implementation without a
    per-update field check.  ``start_from_one`` supports non-standard one-based
    files; repository instances are zero-based.
    """
    with open(file, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"empty instance file: {file}")
    if not lines[0].startswith("#"):
        raise ValueError("instance files must begin with a metadata header")

    metadata = {}
    for token in lines[0][1:].strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            metadata[key] = value
    try:
        num_spins = int(metadata["N"])
    except KeyError as error:
        raise ValueError("instance metadata must specify N") from error
    except ValueError as error:
        raise ValueError("instance metadata N must be an integer") from error
    if num_spins <= 0:
        raise ValueError("instance metadata N must be positive")

    try:
        num_fields = int(metadata.get("num_fields", num_spins))
    except ValueError as error:
        raise ValueError("instance metadata num_fields must be an integer") from error
    if num_fields < 0 or len(lines) - 1 < num_fields:
        raise ValueError("file has fewer field rows than declared in its metadata")

    field_lines = lines[1:1 + num_fields]
    coupling_lines = lines[1 + num_fields:]
    expected_couplings = metadata.get("num_couplings")
    if expected_couplings is not None and len(coupling_lines) != int(expected_couplings):
        raise ValueError(
            f"expected {expected_couplings} coupling rows, found {len(coupling_lines)}"
        )

    index_offset = 1 if start_from_one else 0
    fields = torch.zeros(num_spins, dtype=dtype)
    seen_fields = set()
    for line in field_lines:
        values = line.split()
        if len(values) != 2:
            raise ValueError(f"invalid field row: {line!r}")
        spin = int(values[0]) - index_offset
        if spin < 0 or spin >= num_spins:
            raise ValueError(f"field index out of range: {spin + index_offset}")
        if spin in seen_fields:
            raise ValueError(f"duplicate field for spin {spin + index_offset}")
        seen_fields.add(spin)
        fields[spin] = float(values[1])

    J = torch.zeros((num_spins, num_spins), dtype=dtype)
    seen_edges = set()
    for line in coupling_lines:
        values = line.split()
        if len(values) != 3:
            raise ValueError(f"invalid coupling row: {line!r}")
        first = int(values[0]) - index_offset
        second = int(values[1]) - index_offset
        if first < 0 or first >= num_spins or second < 0 or second >= num_spins:
            raise ValueError(f"coupling index out of range: {line!r}")
        if first == second:
            raise ValueError(f"self-couplings are not supported: {line!r}")
        edge = tuple(sorted((first, second)))
        if edge in seen_edges:
            raise ValueError(f"duplicate coupling for edge {edge}")
        seen_edges.add(edge)
        coupling = float(values[2])
        J[first, second] = coupling
        J[second, first] = coupling

    if device is not None:
        J = J.to(device)
        fields = fields.to(device)
    if return_fields:
        return J, normalize_external_fields(J, fields)
    return J

def schedule_temperatures(Tstart, Tend, num_temps, schedule="linearBeta"):
    """Define a temperature schedule with an explicit number of temperatures."""
    if schedule == "custom":
        temperatures = np.asarray(num_temps)
        if temperatures.ndim != 1 or len(temperatures) == 0:
            raise ValueError("For the custom schedule, num_temps must be a non-empty 1D temperature list")
        return temperatures

    if not isinstance(num_temps, (int, np.integer)) or isinstance(num_temps, bool) or num_temps <= 0:
        raise ValueError("num_temps must be a positive integer")

    if schedule == "linearT":
        temperatures = np.linspace(Tstart, Tend, num_temps)
    elif schedule == "linearBeta":
        temperatures = [1/x for x in np.linspace(1/Tstart, 1/Tend, num_temps)]
    elif schedule == "logT":
        temperatures = np.logspace(np.log10(Tstart), np.log10(Tend), num=num_temps)
    else:
        raise ValueError("Invalid schedule; choose linearT, linearBeta, logT, or custom")
    return temperatures

class Observables:
    """The class of the observables we want to track during training. In this case, we are interested in the mean and minimum energies.
    If we are interested in other observables, we can create use another Observables class.
    
    The observables class needs an observables dictionary to store the values and a method 'update' to update the observables."""
    def __init__(self, J, N, energy_function=compute_energy_zero_field):
        """
        Initialize the Observables class with hard-coded observables.
        """
        self.observables = {
            "min_energy": [],       
            "mean_energy": [],
            "elapsed_time": [],
        }
        self.J = J
        self.N = N
        self.energy_function = energy_function
        self._start_time = None

    def set_start_time(self):
        """Record the wall-clock reference time for elapsed-time tracking."""
        self._start_time = time.time()

    def update(self, population):
        """
        Update the list of observables using the population and temperature.
        
        Parameters:
        population (list or array-like): The population data.
        temperature (float or int): The temperature value.
        """
        energies = self.energy_function(population, self.J, take_mean=False)
        energy_min = energies.min()/self.N
        energy_mean = energies.mean()/self.N
        self.observables["min_energy"].append(float(energy_min))
        self.observables["mean_energy"].append(float(energy_mean))
        if self._start_time is not None:
            self.observables["elapsed_time"].append(time.time() - self._start_time)

    def __repr__(self):
        return f"Observables({self.observables})"

    def get_observable_history(self, observable_name):
        """
        Get the history of a specific observable.
        
        Parameters:
        observable_name (str): The name of the observable to retrieve.
        
        Returns:
        list: The history of the specified observable.
        """
        if observable_name not in self.observables:
            raise ValueError(f"Observable '{observable_name}' not found.")
        
        return self.observables[observable_name]

    def get_elapsed_times(self):
        """Return the list of elapsed times recorded at each update."""
        return self.observables["elapsed_time"]

    def get_running_min_energy(self):
        """Return the cumulative (running) minimum of the per-step min energies."""
        mins = self.observables["min_energy"]
        running = []
        current_min = float("inf")
        for m in mins:
            current_min = min(current_min, m)
            running.append(current_min)
        return running
