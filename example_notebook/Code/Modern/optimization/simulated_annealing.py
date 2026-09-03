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

def simulated_annealing(L, J, pop_size, num_steps_MC, Tstart, Tend, Observables,
                                schedule = "Cv_beta", num_temps_determiner = 0.5, 
                                high_temp_thermalization_steps = 200, dimension = "3d"):

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
    for i in range(high_temp_thermalization_steps):
        population = monte_carlo_update_fast(population, J, beta=1/temperatures[0], even_indices=even_indices, odd_indices=odd_indices)
    observ.update(population) #save the minimum and mean energies


    start_time = time.time()
    observ.set_start_time()
    #perform the simulated annealing
    for T in temperatures[1:]: #we exclude the first temperature, as it was already thermalized
        # Monte Carlo updates
        for i in range(num_steps_MC):
            population = monte_carlo_update_fast(population, J, beta=1/T, even_indices=even_indices, odd_indices=odd_indices)

        observ.update(population)

    end_time = time.time()
    elapsed_time = end_time - start_time
    return temperatures, observ, elapsed_time
