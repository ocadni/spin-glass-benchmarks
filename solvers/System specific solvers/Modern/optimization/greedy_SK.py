
import torch
import numpy as np
import sys
import argparse
import time

sys.path.append("C:/Users/acer/Desktop/stage M1/MLMC_optimization/Code/Legacy/packages")
sys.path.append("C:/Users/acer/Desktop/stage M1/MLMC_optimization/Code/Modern/optimization")
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

def monte_carlo_greedy(pop, J):
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
        acceptance_prob =torch.sign(-delta_E)/2+0.5
        # if the local minimum is reached then no fliping will be accepted
        if torch.sum(acceptance_prob)==0:
            return population,False
        # Apply accepted flips only for accepted positions
        accept=acceptance_prob.bool()
        population[:, indices] = torch.where(accept, proposed_population[:, indices], population[:, indices])

    return population,True

def greedy( J, pop_size, N, Observables,
                                num_temps_determiner = 0.5, batch_size = 256,XORSAT=False):

    #initialize the population
    population = torch.randint(0, 2, (pop_size,N)).float() * 2 - 1
    
    #initialize the observables
    observ = Observables(J, N)
    observ.update(population) #save the minimum and mean energie
    start_time = time.time()
    test=True
    while test:
        population,test = monte_carlo_greedy(population, J)
    observ.update(population)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(population[0])
    return  observ, elapsed_time




#Running the parallel simulated annealing and parsing the arguments
if __name__ == "__main__":
    #parse all the possible arguments
    parser = argparse.ArgumentParser(description='Sequential Tempering')
    #general parsers
    parser.add_argument('--pop_size', type=int, default=1000000, help='Population size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--Tstart', type=float, default=1.92, help='Starting temperature')
    parser.add_argument('--Tend', type=float, default=0.1, help='Ending temperature')
    parser.add_argument('--Cv_factor', type=float, default=1.618, help='Cv_factor')
    parser.add_argument('--MLMCsteps', type=int, default=5, help='Number of Machine-Learning assisted steps')
    parser.add_argument('--MCsteps', type=int, default=15, help='Number of Monte Carlo steps for each MLMC step')
    parser.add_argument('--num_temps', type=int, default=30, help='Number of annealing temperatures')
    parser.add_argument('--schedule', type=str, default="Cv_beta", help='Scheduling of temperatures')
    #specific parsers for training
    parser.add_argument('--num_epochs_start', type=int, default=40, help='Number of epochs for the first training')
    parser.add_argument('--num_epochs_retrain', type=int, default=1, help='Number of epochs for each temperature retraining')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--N', type=int, default=100, help='Number of Machine-Learning assisted steps')

    
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
    seed = args.seed
    Tstart = args.Tstart
    N=args.N
    # .cuda() can be added to the following line
    J = read_couplings(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_SK/Data/Alpha/Couplings/Configurations/SK_N{N}_S{seed}.txt', N)
    '''
    with open(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_SK/Data/Alpha/Couplings/Results/SK_sol_N{N}_S{seed}.txt') as f:
        lines = f.readlines()

        energie = float(lines[0].replace("#", "").strip())
        spins = np.array(list(map(int, lines[1].split())))
    #results=np.loadtxt(f'C:/Users/acer/Desktop/stage M1/MLMC_optimization/Data/Alpha/Couplings/Results/SK_sol_N{N}_S{seed}.txt',skiprows=0)
    '''
    Tstart = float(Tstart)
    Tend = float(Tend)

    MLMCsteps = args.MLMCsteps
    MCsteps = args.MCsteps

    num_epochs_start = args.num_epochs_start
    num_epochs_retrain = args.num_epochs_retrain
    batch_size = args.batch_size


    observ, elapsed_time = greedy( J, pop_size,N,  Observables)
    minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
    print( f"{minimum:.5f}", f"{observ.get_observable_history('mean_energy')[-1]:.5f}", elapsed_time)