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
from device_utils import synchronize

def simulated_annealing(J, pop_size, num_steps_MC, Tstart, Tend, Observables,
                                schedule="linearBeta", num_temps=100,
                                high_temp_thermalization_steps=200, fields=None,
                                return_thermalization_time=False,
                                zero_temperature_quench=False):
    """Run simulated annealing.

    Returns ``(temperatures, observables, elapsed_time)``, where
    ``elapsed_time`` covers the annealing loop only.  With
    ``return_thermalization_time=True`` the wall-clock time of the
    high-temperature thermalization sweeps is appended as a fourth element.
    With ``zero_temperature_quench=True`` the final population
    is quenched with ``run_zero_temperature_quench`` (after the timers stop,
    so the times above exclude it) and its summary dict is appended last.
    The observables history is not changed by the quench.
    """

    device = J.device

    # The coupling graph determines both the number of spins and whether a
    # checkerboard update is valid.
    N = get_num_spins(J)
    energy_function, fields = select_energy_function(J, fields)
    mc_update, even_indices, odd_indices = select_monte_carlo_update(J, fields)

    #set the temperature schedule
    temperatures = schedule_temperatures(Tstart, Tend, num_temps, schedule)

    #initialize the population
    population = torch.randint(0, 2, (pop_size,N), device=device).float() * 2 - 1
    
    #initialize the observables
    observ = Observables(J, N, energy_function=energy_function)

    # Thermalize the high temperature population
    synchronize(device)
    thermalization_start_time = time.time()
    for i in range(high_temp_thermalization_steps):
        population = mc_update(population, J, beta=1/temperatures[0], even_indices=even_indices, odd_indices=odd_indices)
    observ.update(population) #save the minimum and mean energies


    start_time = time.time()
    observ.set_start_time()
    #perform the simulated annealing
    for T in temperatures[1:]: #we exclude the first temperature, as it was already thermalized
        # Monte Carlo updates
        for i in range(num_steps_MC):
            population = mc_update(population, J, beta=1/T, even_indices=even_indices, odd_indices=odd_indices)

        observ.update(population)

    end_time = time.time()
    elapsed_time = end_time - start_time
    results = (temperatures, observ, elapsed_time)
    if return_thermalization_time:
        results += (start_time - thermalization_start_time,)
    if zero_temperature_quench:
        results += (run_zero_temperature_quench(
            population, J, energy_function, fields=fields,
            even_indices=even_indices, odd_indices=odd_indices,
        ),)
    return results
