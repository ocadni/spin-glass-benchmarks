import torch
import numpy as np
import sys
import argparse
import time
from functools import partial

sys.path.append("../../../Code/Legacy/packages")
sys.path.append("../../../Code/Modern/optimization")

from geometry import *
from utilities import *
from data_loads import *
from monte_carlo import *
from made import *
from global_steps import *
from device_utils import empty_cache, synchronize

from copy import deepcopy
import time

def _MLMC_fast(model, data, beta, N, J, energy_function, num_steps=10,
               return_correlations=False, mini_batch_size=20_000):
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

            current_energy = energy_function(current_config, J)
            new_energy = energy_function(new_config, J)

            arg_new = -beta * new_energy + all_new_probabilities
            arg_current = -beta * current_energy + all_current_probabilities

            acceptances = (torch.log(torch.rand(size=(len(data),), device=data.device)) < (arg_new - arg_current)).int()
            current_config = torch.einsum("i, ij->ij", (1 - acceptances), current_config) + torch.einsum("i, ij->ij", acceptances, new_config)

            empty_cache(data.device)

            acc_rates.append(torch.sum(acceptances) / len(data))


            if return_correlations:
                correlations.append(float(torch.mean(data * current_config) - torch.mean(data) * torch.mean(current_config)))

    if return_correlations:
        return current_config, acc_rates, correlations
    else:
        return current_config, acc_rates


def MLMC_fast_zero_field(model, data, beta, N, J, num_steps=10,
                         return_correlations=False, mini_batch_size=20_000):
    """Global Metropolis update for the original pairwise-only energy."""
    return _MLMC_fast(
        model, data, beta, N, J, compute_energy_zero_field, num_steps,
        return_correlations, mini_batch_size,
    )


def MLMC_fast_with_fields(model, data, beta, N, J, fields, num_steps=10,
                          return_correlations=False, mini_batch_size=20_000):
    """Global Metropolis update for ``-1/2 s^T J s - h^T s``."""
    energy_function, fields = select_energy_function(J, fields)
    if fields is None:
        return MLMC_fast_zero_field(
            model, data, beta, N, J, num_steps, return_correlations, mini_batch_size
        )
    return _MLMC_fast(
        model, data, beta, N, J, energy_function, num_steps,
        return_correlations, mini_batch_size,
    )


# Backward-compatible name for direct callers of the original zero-field path.
MLMC_fast = MLMC_fast_zero_field
    

def global_annealing(J, pop_size, num_steps_MC, swap_step, Tstart, Tend, Observables,
                                schedule="linearBeta", num_temps=100,
                                high_temp_thermalization_steps=200, batch_size=256,
                                num_epochs_start=40, num_epochs_retrain=1,
                                fields=None):


    device = J.device

    # The coupling graph determines both the number of spins and whether a
    # checkerboard update is valid.
    N = get_num_spins(J)
    energy_function, fields = select_energy_function(J, fields)
    mc_update, even_indices, odd_indices = select_monte_carlo_update(J, fields)
    if fields is None:
        mlmc_update = MLMC_fast_zero_field
        train_model = train_made_improved_zero_field
        retrain_model = retrain_made_zero_field
    else:
        mlmc_update = partial(MLMC_fast_with_fields, fields=fields)
        train_model = train_made_improved_with_fields
        retrain_model = retrain_made_with_fields

    #set the temperature schedule
    temperatures = schedule_temperatures(Tstart, Tend, num_temps, schedule)

    #initialize the population
    population = torch.randint(0, 2, (pop_size,N), device=device).float() * 2 - 1
    
    #initialize the observables
    observ = Observables(J, N, energy_function=energy_function)

    # Thermalize the high temperature population
    oldT = temperatures[0]
    for i in range(high_temp_thermalization_steps):
        population = mc_update(population, J, beta=1/oldT, even_indices=even_indices, odd_indices=odd_indices)
    observ.update(population) #save the minimum and mean energie
    synchronize(device)
    start_time_1 = time.time()
    model = train_model(population, N, epochs=num_epochs_start)
    synchronize(device)
    start_time_2 = time.time()
    observ.set_start_time()
    for currT in temperatures[1:-1]:
        #Monte Carlo updates
        for i in range(num_steps_MC):
            population, _ = mlmc_update(model, population, 1/currT, N, J, num_steps=1)
            for j in range(swap_step):
                population = mc_update(population, J, 1/currT, even_indices, odd_indices)
        #retrain of the model
        model = retrain_model(
            model, population, epochs=num_epochs_retrain, batch_size=batch_size
        )
        observ.update(population)

    #for the last temperature, we do not need to retrain the model
    currT = temperatures[-1]
    for i in range(num_steps_MC):
        population, _ = mlmc_update(model, population, 1/currT, N, J, num_steps=1)
        for j in range(swap_step):
            population = mc_update(population, J, 1/currT, even_indices, odd_indices)
    observ.update(population)
    synchronize(device)
    end_time = time.time()

    elapsed_train_time = start_time_2 - start_time_1
    elapsed_annealing_time = end_time - start_time_2
    return temperatures, observ, elapsed_train_time, elapsed_annealing_time
