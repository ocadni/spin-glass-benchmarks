#Code to perform standard Metropolis MC, with optimization for parallelization
#In particular, we use a checkerboard update, where we update even and odd indices separately

import torch
import numpy as np
import sys

sys.path.append("../../../Code/Legacy/packages")
from utilities import get_betas_3d, compute_energy





def monte_carlo_update_fast(pop, J, beta):
    """Monte Carlo update model using a checkerboard pattern."""
    population = pop.clone()
    pop_size, N = population.shape
    # Define "even" and "odd" indices for a checkerboard update

    # Update spins in two passes (checkerboard pattern)
    for indices in range(N):
        # Propose flips for the entire population at selected indices
        proposed_population = population.clone()
        proposed_population[:, indices] *= -1
        
        # Compute energy difference for each single-spin flip
        delta_E = -2* torch.einsum("i, i->i",proposed_population[:, indices], torch.einsum("kj, j->k", population, J[indices, :]))
        
        # Metropolis acceptance criterion for each spin
        acceptance_prob = torch.exp(-beta * delta_E)
        random_vals = torch.rand(pop_size, device=population.device)
        accept = (delta_E < 0) | (random_vals < acceptance_prob)
        
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
    return J

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
    def __init__(self, J, N):
        """
        Initialize the Observables class with hard-coded observables.
        """
        self.observables = {
            "min_energy": [],       
            "mean_energy": [],
            "energy": [],
        }
        self.J = J
        self.N = N

    def update(self, population):
        """
        Update the list of observables using the population and temperature.
        
        Parameters:
        population (list or array-like): The population data.
        temperature (float or int): The temperature value.
        """
        energies = compute_energy(population, self.J, take_mean=False)
        energy_min = energies.min()/self.N
        energy_mean = energies.mean()/self.N
        self.observables["min_energy"].append(float(energy_min))
        self.observables["mean_energy"].append(float(energy_mean)) 
        # for p sucess measure
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