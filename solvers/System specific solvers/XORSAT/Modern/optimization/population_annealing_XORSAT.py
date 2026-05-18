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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def systematic_resampling(probabilities: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Perform systematic resampling of particles based on their probabilities.
    
    Args:
    - probabilities (torch.Tensor): A 1D tensor of probabilities of shape (N,). 
                                    The probabilities should be normalized to sum to 1.
    - num_samples (int): Number of samples to draw.
    
    Returns:
    - indices (torch.Tensor): The indices of resampled particles.
    # Move probabilities to GPU if not already there
    if not probabilities.is_cuda:
        probabilities = probabilities.cuda()
    """
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

def population_annealing(K, J,h, pop_size, num_steps_MC, Tstart, Tend, Observables, 
                                schedule = "Cv_beta", num_temps_determiner = 0.5, 
                                high_temp_thermalization_steps = 200,
                                reweight_mode = "multinomial"):
    #num_temps_determiner is either Cv_factor or the number of temperatures depending on the schedule

    #get the indices (needed for the checkerboard update)
    N=2*K
    INDICES=greedy_independent_sets(J)[0]
    #set the temperature schedule
    temperatures = schedule_temperatures(Tstart, Tend, num_temps_determiner, schedule, N)

    #initialize the population
    population = torch.randint(0, 2, (pop_size,N)).float().to(device) * 2 - 1
    
    #initialize the observables
    observ = Observables(J, N,h)

    # Thermalize the high temperature population
    oldT = temperatures[0]
    for i in range(high_temp_thermalization_steps):
        population = monte_carlo_update_fast(population, J,h, 1/oldT, INDICES)
    observ.update(population) #save the minimum and mean energies


    start_time = time.time()
    #perform the simulated annealing
    for T in temperatures[1:]: #we exclude the first temperature, as it was already thermalized

        # Perform the lowering themperature step
        DeltaB = 1/T - 1/oldT
        energies = compute_energy(population, J,h)
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
            population = monte_carlo_update_fast(population, J,h, 1/T, INDICES)
        observ.update(population)
    
        # Set T as oldT
        oldT = T
    end_time = time.time()
    elapsed_time = end_time - start_time
    return temperatures, observ, elapsed_time


#Running the parallel population annealing and parsing the data
if __name__ == "__main__":
    #parse all the possible arguments
    parser = argparse.ArgumentParser(description='Population Annealing')
    parser.add_argument('--pop_size', type=int, default=1000, help='Population size')
    parser.add_argument('--K', type=int, default=999, help='Lattice size')
    parser.add_argument('--seed', type=int, default=345692, help='Random seed')
    parser.add_argument('--Tstart', type=float, default=1.92, help='Starting temperature')
    parser.add_argument('--Tend', type=float, default=0.1, help='Ending temperature')
    parser.add_argument('--Cv_factor', type=float, default=1.618, help='Cv_factor')
    parser.add_argument('--MCsteps', type=int, default=320, help='Number of Monte Carlo steps')
    parser.add_argument('--num_temps', type=int, default=30, help='Number of annealing temperatures')
    parser.add_argument('--schedule', type=str, default="linearT", help='Scheduling of temperatures')
    args = parser.parse_args()

    #check consistency for the schedules: Cv_beta requires Cv_factor, the others the number of steps
    if args.schedule != "Cv_beta" and args.Cv_factor != parser.get_default("Cv_factor"):
        parser.error("Cv_factor can only be specified when schedule is Cv_beta.")
    if args.schedule == "Cv_beta" and args.num_temps != parser.get_default("num_temps"):
        parser.error("num_temps cannot be specified when schedule is Cv_beta.")

    if args.schedule == "Cv_beta":
        num_temps_determiner = args.Cv_factor
    else:    
        num_temps_determiner = args.num_temps

    #define the parameters of the model
    pop_size = args.pop_size
    Tend = args.Tend
    Cv_factor = args.Cv_factor
    schedule = args.schedule
    K = args.K
    N = K*2
    seed = args.seed
    Tstart = args.Tstart
    J = read_couplings(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Data/Alpha/Couplings/couplings_XORSAT_K{K}_seed{seed}.txt', N).to(device)
    h=read_field(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Data/Alpha/Couplings/field_XORSAT_K{K}_seed{seed}.txt', N).to(device)
    Tstart = float(Tstart)
    Tend = float(Tend)

    MCsteps = args.MCsteps

    #perform the simulated annealing
    temperatures, observ, elapsed_time = population_annealing(K, J,h, pop_size, MCsteps, Tstart, Tend, Observables, num_temps_determiner = num_temps_determiner, schedule=schedule)
    minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
    real_minimum = torch.tensor(observ.get_observable_history("min_real_energy")).min()
    print( f"{minimum:.5f}", f"{observ.get_observable_history('mean_energy')[-1]:.5f}", elapsed_time)
    print( f"{real_minimum:.5f}", f"{observ.get_observable_history('mean_real_energy')[-1]:.5f}", elapsed_time)
    print(MCsteps, f"{len(temperatures)}", schedule, f"{minimum:.5f}", f'{observ.get_observable_history("mean_energy")[-1]:.5f}', elapsed_time)