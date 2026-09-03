import torch
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

def systematic_resampling(probabilities: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Perform systematic resampling of particles based on their probabilities.
    
    Args:
    - probabilities (torch.Tensor): A 1D tensor of probabilities of shape (N,). 
                                    The probabilities should be normalized to sum to 1.
    - num_samples (int): Number of samples to draw.
    
    Returns:
    - indices (torch.Tensor): The indices of resampled particles.
    """
    # Move probabilities to GPU if not already there
    if not probabilities.is_cuda:
        probabilities = probabilities.cuda()
    
    N = probabilities.size(0)  # Number of particles
    
    # Compute the cumulative sum of probabilities (CDF)
    cumulative_sum = torch.cumsum(probabilities, dim=0)
    
    # Generate a random starting point in the interval [0, 1/num_samples)
    start = torch.rand(1, device=probabilities.device) / num_samples
    
    # Create the positions where we will sample from the CDF
    positions = start + torch.arange(num_samples, dtype=torch.float32, device=probabilities.device) / num_samples
    
    # Find the indices corresponding to the positions in the cumulative sum
    indices = torch.searchsorted(cumulative_sum, positions)
    
    return indices

def population_annealing(L, J, pop_size, num_steps_MC, Tstart, Tend, Observables, 
                                schedule = "Cv_beta", num_temps_determiner = 0.5, 
                                high_temp_thermalization_steps = 200, dimension = "3d",
                                reweight_mode = "multinomial"):
    #num_temps_determiner is either Cv_factor or the number of temperatures depending on the schedule

    #get the indices (needed for the checkerboard update)
    if dimension == "3d":
        even_indices, odd_indices = get_indices(L)
        N = L*L*L
    elif dimension == "2d":
        even_indices, odd_indices = get_indices_2D(L)
        N = L*L
    else:
        raise ValueError("dimension must be either 3d or 2d")
    
    #set the temperature schedule
    temperatures = schedule_temperatures(Tstart, Tend, num_temps_determiner, schedule, N)

    #initialize the population
    population = torch.randint(0, 2, (pop_size,N), device="cuda").float() * 2 - 1
    
    #initialize the observables
    observ = Observables(J, N)

    # Thermalize the high temperature population
    oldT = temperatures[0]
    for i in range(high_temp_thermalization_steps):
        population = monte_carlo_update_fast(population, J, beta=1/oldT, even_indices=even_indices, odd_indices=odd_indices)
    observ.update(population) #save the minimum and mean energies


    start_time = time.time()
    observ.set_start_time()
    #perform the simulated annealing
    for T in temperatures[1:]: #we exclude the first temperature, as it was already thermalized

        # Perform the lowering themperature step
        DeltaB = 1/T - 1/oldT
        energies = compute_energy(population, J)
        energies = energies - torch.min(energies)
        #weights = torch.exp(-energies*DeltaB)
        probabilities = torch.softmax(-energies * DeltaB, dim=0)
        if reweight_mode == "multinomial":
            resampled_indices = torch.multinomial(probabilities, num_samples=pop_size, replacement=True)
        elif reweight_mode == "systematic":
            resampled_indices = systematic_resampling(probabilities, num_samples=pop_size) 
        else:
            raise ValueError(f"reweight_mode {reweight_mode} not supported")
        resampled_population = population[resampled_indices]
        population = resampled_population

        # Monte Carlo updates
        for i in range(num_steps_MC):
            population = monte_carlo_update_fast(population, J, beta=1/T, even_indices=even_indices, odd_indices=odd_indices)
        observ.update(population)
    
        # Set T as oldT
        oldT = T
    end_time = time.time()
    elapsed_time = end_time - start_time
    return temperatures, observ, elapsed_time
