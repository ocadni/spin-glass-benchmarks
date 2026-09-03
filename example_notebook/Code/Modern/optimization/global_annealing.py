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
from made import *
from global_steps import *

from copy import deepcopy
import time

def MLMC_fast(model, data, beta, N, J, num_steps=10, return_correlations=False, mini_batch_size=20_000):
    """
    Performs the Metropolis-Langevin Monte Carlo update step, processing the
    large population in smaller mini-batches to avoid out-of-memory errors.
    """
    acc_rates = []
    if return_correlations:
        correlations = [1]

    with torch.no_grad():
        bce = nn.BCELoss(reduction="none")
        current_config = data.clone()

        for t in range(num_steps):

            new_config = generate_config_fast(model, N, len(data), J)

            # --- BATCHING LOGIC START ---
            all_current_probabilities = torch.empty(len(data), device=data.device)
            all_new_probabilities = torch.empty(len(data), device=data.device)

            for i in range(0, len(data), mini_batch_size):

                current_batch = current_config[i:i+mini_batch_size]
                new_batch = new_config[i:i+mini_batch_size]

                current_probs_output = model(current_batch)
                new_probs_output = model(new_batch)

                current_probs_batch = torch.sum(bce(current_probs_output, (current_batch + 1) / 2), axis=1) #this gives the negative log-probability
                new_probs_batch = torch.sum(bce(new_probs_output, (new_batch + 1) / 2), axis=1)

                all_current_probabilities[i:i+mini_batch_size] = current_probs_batch
                all_new_probabilities[i:i+mini_batch_size] = new_probs_batch
            # --- BATCHING LOGIC END ---

            current_energy = compute_energy(current_config, J)
            new_energy = compute_energy(new_config, J)

            arg_new = -beta * new_energy + all_new_probabilities
            arg_current = -beta * current_energy + all_current_probabilities

            acceptances = (torch.log(torch.rand(size=(len(data),), device="cuda")) < (arg_new - arg_current)).int()
            current_config = torch.einsum("i, ij->ij", (1 - acceptances), current_config) + torch.einsum("i, ij->ij", acceptances, new_config)

            torch.cuda.empty_cache()

            acc_rates.append(torch.sum(acceptances) / len(data))


            if return_correlations:
                correlations.append(float(torch.mean(data * current_config) - torch.mean(data) * torch.mean(current_config)))

    if return_correlations:
        return current_config, acc_rates, correlations
    else:
        return current_config, acc_rates
    

def global_annealing(L, J, pop_size, num_steps_MC, swap_step, Tstart, Tend, Observables,
                                schedule = "Cv_beta", num_temps_determiner = 0.5, 
                                high_temp_thermalization_steps = 200, batch_size = 256, 
                                num_epochs_start = 40, num_epochs_retrain = 1, dimension = "3d"):


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
    observ.update(population) #save the minimum and mean energie
    torch.cuda.synchronize()
    start_time_1 = time.time()
    model = train_made_improved(population, N, epochs = num_epochs_start)
    torch.cuda.synchronize()
    start_time_2 = time.time()
    observ.set_start_time()
    for currT in temperatures[1:-1]:
        #Monte Carlo updates
        for i in range(num_steps_MC):
            population, _ = MLMC_fast(model, population, 1/currT, N, J, num_steps=1)
            for j in range(swap_step):
                population = monte_carlo_update_fast(population, J, 1/currT, even_indices, odd_indices)
        #retrain of the model
        model = retrain_made(model, population, epochs = num_epochs_retrain, batch_size=batch_size)
        observ.update(population)

    #for the last temperature, we do not need to retrain the model
    currT = temperatures[-1]
    for i in range(num_steps_MC):
        population, _ = MLMC_fast(model, population, 1/currT, N, J, num_steps=1)
        for j in range(swap_step):
            population = monte_carlo_update_fast(population, J, 1/currT, even_indices, odd_indices)
    observ.update(population)
    torch.cuda.synchronize()
    end_time = time.time()

    elapsed_train_time = start_time_2 - start_time_1
    elapsed_annealing_time = end_time - start_time_2
    return temperatures, observ, elapsed_train_time, elapsed_annealing_time

