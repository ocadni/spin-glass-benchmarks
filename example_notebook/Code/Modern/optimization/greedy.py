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


def greedy_search(L, J, pop_size, num_steps, Observables, dimension="3d"):
    """
    Runs a greedy optimization (Zero-Temperature Monte Carlo).
    Only moves that decrease or maintain the energy are accepted.
    """

    #get the indices (needed for the checkerboard update)
    if dimension == "3d":
        even_indices, odd_indices = get_indices(L)
        N = L*L*L
    elif dimension == "2d":
        even_indices, odd_indices = get_indices_2D(L)
        N = L*L
    else:
        raise ValueError("dimension must be either 3d or 2d")

    #initialize the population
    population = torch.randint(0, 2, (pop_size, N), device="cuda").float() * 2 - 1
    
    #initialize the observables
    observ = Observables(J, N)
    
    #set beta to infinity for greedy behavior
    beta_greedy = float('inf')

    start_time = time.time()
    observ.set_start_time()

    # perform the Greedy Search
    for i in range(num_steps):
        population = monte_carlo_update_fast(
            population, 
            J, 
            beta=beta_greedy, 
            even_indices=even_indices, 
            odd_indices=odd_indices
        )
        
        # Record energy/observables after each update
        observ.update(population)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return observ, elapsed_time