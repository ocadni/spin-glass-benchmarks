#Code to perform standard Metropolis MC, with optimization for parallelization
#In particular, we use a checkerboard update, where we update even and odd indices separately

import torch
import numpy as np
import sys

#sys.path.append("../../../Code/Legacy/packages")
sys.path.append("C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Code/Legacy/packages")
sys.path.append("C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Code/Modern/optimization")
from utilities import get_betas_3d, compute_energy, compute_real_energy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_indices(dim=10):
    """
    Generate even and odd indices tensors for a 3D checkerboard pattern in a dim x dim x dim cube.

    Parameters:
    dim (int): The dimension of the cube. Default is 10 for a 10x10x10 cube.

    Returns:
    tuple: A tuple containing two numpy arrays:
           - even_indices_tensor: Indices for even (checkerboard) cells
           - odd_indices_tensor: Indices for odd (checkerboard) cells
    """
    # Initialize lists for even and odd indices
    even_indices = []
    odd_indices = []

    # Loop through each (x, y, z) coordinate in the cube
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                # Calculate the 1D index from the (x, y, z) coordinates
                index = x * dim * dim + y * dim + z

                # Check the checkerboard condition and append to appropriate list
                if (x + y + z) % 2 == 0:
                    even_indices.append(index)
                else:
                    odd_indices.append(index)

    # Convert lists to tensors (numpy arrays)
    even_indices_tensor = torch.tensor(even_indices)
    odd_indices_tensor = torch.tensor(odd_indices)

    return even_indices_tensor, odd_indices_tensor





def get_indices_2D(dim=10):
    """
    Generate even and odd indices tensors for a 2D checkerboard pattern in a dim x dim grid.

    Parameters:
    dim (int): The dimension of the grid. Default is 10 for a 10x10 grid.

    Returns:
    tuple: A tuple containing two tensors (CUDA-compatible):
           - even_indices_tensor: Indices for even (checkerboard) cells
           - odd_indices_tensor: Indices for odd (checkerboard) cells
    """
    # Initialize lists for even and odd indices
    even_indices = []
    odd_indices = []

    # Loop through each (x, y) coordinate in the grid
    for x in range(dim):
        for y in range(dim):
            # Calculate the 1D index from the (x, y) coordinates
            index = x * dim + y

            # Check the checkerboard condition and append to the appropriate list
            if (x + y) % 2 == 0:
                even_indices.append(index)
            else:
                odd_indices.append(index)

    # Convert lists to tensors (CUDA-compatible if available)
    even_indices_tensor = torch.tensor(even_indices)
    odd_indices_tensor = torch.tensor(odd_indices)

    return even_indices_tensor, odd_indices_tensor


def greedy_independent_sets(J):
    """
    Partition the indices of a connection matrix into independent sets
    using a greedy algorithm.
 
    An independent set is a group of indices where no two are connected
    (i.e., J[i][j] == 0 for all pairs i, j within the set).
 
    Parameters
    ----------
    J : array-like, shape (n, n)
        Connection matrix. J[i][j] != 0 means i and j interact.
 
    Returns
    -------
    sets : list of list of int
        Each inner list is an independent set of indices.
    assignment : list of int
        assignment[i] is the set index that index i was placed in.
    """
    J = np.array(J)
    n = J.shape[0]
    sets = []        # list of sets (stored as sets for O(1) membership check)
    assignment = [-1] * n
 
    for i in range(n):
        # Find neighbours of i (all j where J[i][j] != 0 or J[j][i] != 0)
        neighbours = set(
            j for j in range(n)
            if j != i and (J[i][j] != 0 or J[j][i] != 0)
        )
 
        placed = False
        for set_idx, current_set in enumerate(sets):
            # i can join this set only if none of its neighbours are already in it
            if current_set.isdisjoint(neighbours):
                current_set.add(i)
                assignment[i] = set_idx
                placed = True
                break
 
        if not placed:
            # No existing set works — open a new one
            sets.append({i})
            assignment[i] = len(sets) - 1
 
    # Convert sets to sorted lists for readability
    return [sorted(s) for s in sets], torch.tensor(assignment)
 
 

 
# ── validation helper ─────────────────────────────────────────────────────────
 
def validate(J, sets):
    """Check that no two indices in the same set are connected."""
    J = np.array(J)
    for k, s in enumerate(sets):
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                i, j = s[a], s[b]
                if J[i][j] != 0 or J[j][i] != 0:
                    print(f"  ✗ Conflict in Set {k}: indices {i} and {j} interact!")
                    return False
    print("  ✓ All sets are valid independent sets.")
    return True





def monte_carlo_update_fast(pop, J,h, beta, INDICES):
    """Monte Carlo update model using a checkerboard pattern."""
    population = pop.clone()
    pop_size, N = population.shape
    # Define "even" and "odd" indices for a checkerboard update
    #INDICES=greedy_independent_sets(J)[0]
    # Update spins in two passes (checkerboard pattern)
    for indices in INDICES:
        # Propose flips for the entire population at selected indices
        proposed_population = population.clone()
        proposed_population[:, indices] *= -1
        
        # Compute energy difference for each single-spin flip
        delta_E = -2* torch.einsum("ki, ki->ki",proposed_population[:, indices], torch.einsum("kj, ji->ki", population, J[indices, :].T)+h[indices])
        
        # Metropolis acceptance criterion for each spin
        #acceptance_prob = torch.exp(-beta * delta_E)
        #random_vals = torch.rand(pop_size, len(indices), device=population.device)
        #accept = (delta_E < 0) | (random_vals < acceptance_prob)
        
        log_rand = torch.log(torch.rand_like(delta_E)).to(device)
        accept   = (delta_E < 0) | (log_rand < -beta * delta_E)              # [pop, |idx|]
        
        
        # Apply accepted flips only for accepted positions
        population[:, indices] = torch.where(accept, proposed_population[:, indices], population[:, indices])

    return population





def read_couplings(file, N, start_from_one = False):
    """Read couplings from a file and return as a symmetric matrix."""
    x = np.loadtxt(file,skiprows=N+1)
    if start_from_one:
        x[:, 0] = x[:, 0] - 1
        x[:, 1] = x[:, 1] - 1
    J = torch.zeros(N,N)
    for i in range(x.shape[0]):
        a, b, value = int(x[i, 0]), int(x[i, 1]), x[i, 2]
        J[a, b] = value
        J[b, a] = value
    x = np.loadtxt(file,skiprows=1,max_rows=N)
    if start_from_one:
        x[:, 0] = x[:, 0] - 1
    h = torch.zeros(N)
    for i in range(x.shape[0]):
        a, value = int(x[i, 0]), x[i, 1]
        h[a] = value
    return J,h


def schedule_temperatures(Tstart, Tend, num_temps_determiner, schedule, N):
    "Define the schedule of temperatures"
    if schedule == "Cv_beta":
        temperatures = [1/x for x in get_betas_3d(1/Tstart, 1/Tend, num_temps_determiner, N)] # schedule based on Cv
    elif schedule == "linearT":
        temperatures = np.linspace(Tstart, Tend, num_temps_determiner)
    elif schedule == "linearBeta":
        temperatures = [1/x for x in np.linspace(1/Tstart, 1/Tend, num_temps_determiner)]
    elif schedule == "logT":
        temperatures = np.logspace(np.log10(Tstart), np.log10(Tend), num=num_temps_determiner)
    elif schedule == "custom": #In this case I assume num_temps_determiner is the list of temperatures
        temperatures = num_temps_determiner
    else:
        raise ValueError("Invalid schedule")
    return temperatures

class Observables:
    """The class of the observables we want to track during training. In this case, we are interested in the mean and minimum energies.
    If we are interested in other observables, we can create use another Observables class.
    
    The observables class needs an observables dictionary to store the values and a method 'update' to update the observables."""
    def __init__(self, J, N,h):
        """
        Initialize the Observables class with hard-coded observables.
        """
        self.observables = {
            "min_energy": [],       
            "mean_energy": [],
            "min_real_energy": [],       
            "mean_real_energy": [],
            "energy": [],
            
        }
        self.J = J
        self.N = N
        self.h=h

    def update(self, population):
        """
        Update the list of observables using the population and temperature.
        
        Parameters:
        population (list or array-like): The population data.
        temperature (float or int): The temperature value.
        """
        energies = compute_energy(population, self.J,self.h, take_mean=False)
        energy_min = energies.min()/self.N
        energy_mean = energies.mean()/self.N
        real_energies = compute_real_energy(population, self.J,self.h, take_mean=False)
        # the factor *2 is here because N//2 are only the real spins
        real_energy_min = real_energies.min()/self.N*2
        real_energy_mean = real_energies.mean()/self.N*2
        self.observables["min_energy"].append(float(energy_min))
        self.observables["mean_energy"].append(float(energy_mean))   
        self.observables["min_real_energy"].append(float(real_energy_min))
        self.observables["mean_real_energy"].append(float(real_energy_mean)) 
        self.observables["energy"].append(energies)
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