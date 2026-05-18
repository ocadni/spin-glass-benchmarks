
import torch
import numpy as np
import sys
import argparse
import time

sys.path.append("C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Code/Legacy/packages")
sys.path.append("C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Code/Modern/optimization")

from geometry import *
from utilities import *
from data_loads import *
from monte_carlo import *
from made import *
from global_steps import *

from copy import deepcopy
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def monte_carlo_greedy(pop, J,h, INDICES):
    """Monte Carlo update model using a checkerboard pattern."""
    population = pop.clone()
    pop_size, N = population.shape
    
    # Update spins in two passes (checkerboard pattern)
    for indices in INDICES:
        # Propose flips for the entire population at selected indices
        proposed_population = population.clone()
        proposed_population[:, indices] *= -1
        
        # Compute energy difference for each single-spin flip
        #delta_E = -2* torch.einsum("ki, ki->ki",proposed_population[:, indices], torch.einsum("kj, ji->ki", population, J[indices, :].T)+h[indices])
        local_field = torch.einsum("kj,ji->ki", population, J[indices, :].T) + h[indices]
        delta_E = 2 * population[:, indices] * local_field
        
        # Metropolis acceptance criterion for each spin
        acceptance_prob =torch.sign(-delta_E)/2+0.5
        # if the local minimum is reached then no fliping will be accepted
        if torch.sum(acceptance_prob)==0:
            return population,False
        # Apply accepted flips only for accepted positions
        accept=acceptance_prob.bool()
        population[:, indices] = torch.where(accept, proposed_population[:, indices], population[:, indices])

    return population,True

def greedy(K, J,h, pop_size, N, Observables,
                                num_temps_determiner = 0.5, batch_size = 256):

    #get the indices (needed for the checkerboard update)
    INDICES=greedy_independent_sets(J)[0]
    """
    if dimension == "3d":
        N = L*L*L*2
    elif dimension == "2d":
        N = L*L*2
    else:
        raise ValueError("dimension must be either 3d or 2d")
    """
    N=K*2
    #initialize the population
    population = torch.randint(0, 2, (pop_size,N)).float() * 2 - 1
    
    #initialize the observables
    observ = Observables(J, N,h)
    observ.update(population) #save the minimum and mean energie
    start_time = time.time()
    test=True
    while test:
        population,test = monte_carlo_greedy(population, J,h, INDICES)
    observ.update(population)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(population[0])
    print(compute_one_energy(population[0],J,h))
    return  observ, elapsed_time




#Running the parallel simulated annealing and parsing the arguments
if __name__ == "__main__":
    #parse all the possible arguments
    parser = argparse.ArgumentParser(description='Sequential Tempering')
    #general parsers
    parser.add_argument('--pop_size', type=int, default=10000, help='Population size')
    parser.add_argument('--K', type=int, default=7, help='Lattice size')
    parser.add_argument('--seed', type=int, default= 345692, help='Random seed')
    parser.add_argument('--Tstart', type=float, default=1.92, help='Starting temperature')
    parser.add_argument('--Tend', type=float, default=0.1, help='Ending temperature')
    parser.add_argument('--Cv_factor', type=float, default=1.618, help='Cv_factor')
    parser.add_argument('--MLMCsteps', type=int, default=5, help='Number of Machine-Learning assisted steps')
    parser.add_argument('--MCsteps', type=int, default=15, help='Number of Monte Carlo steps for each MLMC step')
    parser.add_argument('--num_temps', type=int, default=30, help='Number of annealing temperatures')
    parser.add_argument('--schedule', type=str, default="Cv_beta", help='Scheduling of temperatures')
    parser.add_argument('--dimension', type=str, default="2d", help='dimension')
    #specific parsers for training
    parser.add_argument('--num_epochs_start', type=int, default=40, help='Number of epochs for the first training')
    parser.add_argument('--num_epochs_retrain', type=int, default=1, help='Number of epochs for each temperature retraining')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    
    
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
    dim=int(args.dimension[0])
    N = K*2
    seed = args.seed
    Tstart = args.Tstart
    # .cuda() can be added to the following line
    J = read_couplings(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Data/Alpha/Couplings/couplings_XORSAT_K{K}_seed{seed}.txt', N).to(device)
    h=read_field(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Data/Alpha/Couplings/field_XORSAT_K{K}_seed{seed}.txt', N).to(device)
    Tstart = float(Tstart)
    Tend = float(Tend)

    MLMCsteps = args.MLMCsteps
    MCsteps = args.MCsteps

    num_epochs_start = args.num_epochs_start
    num_epochs_retrain = args.num_epochs_retrain
    batch_size = args.batch_size


    observ, elapsed_time = greedy(K, J,h, pop_size,N,  Observables)
    minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
    real_minimum = torch.tensor(observ.get_observable_history("min_real_energy")).min()
    print( f"{minimum:.5f}", f"{observ.get_observable_history('mean_energy')[-1]:.5f}", elapsed_time)
    print( f"{real_minimum:.5f}", f"{observ.get_observable_history('mean_real_energy')[-1]:.5f}", elapsed_time)