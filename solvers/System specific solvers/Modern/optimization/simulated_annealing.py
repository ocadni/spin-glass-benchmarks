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

def simulated_annealing( J, pop_size, num_steps_MC, N, Tstart, Tend, Observables,
                                schedule = "Cv_beta", num_temps_determiner = 0.5, 
                                high_temp_thermalization_steps = 200, dimension = "3d"):



    #set the temperature schedule
    temperatures = schedule_temperatures(Tstart, Tend, num_temps_determiner, schedule, N)

    #initialize the population
    population = torch.randint(0, 2, (pop_size,N)).float() * 2 - 1
    
    #initialize the observables
    observ = Observables(J, N)

    # Thermalize the high temperature population
    for i in range(high_temp_thermalization_steps):
        population = monte_carlo_update_fast(population, J, beta=1/temperatures[0])
    observ.update(population) #save the minimum and mean energies


    start_time = time.time()
    #perform the simulated annealing
    for T in temperatures[1:]: #we exclude the first temperature, as it was already thermalized
        # Monte Carlo updates
        for i in range(num_steps_MC):
            population = monte_carlo_update_fast(population, J, beta=1/T)

        observ.update(population)

    end_time = time.time()
    elapsed_time = end_time - start_time
    return temperatures, observ, elapsed_time


#Running the parallel simulated annealing and parsing the arguments
if __name__ == "__main__":
    #parse all the possible arguments
    parser = argparse.ArgumentParser(description='Simulated Annealing Fast')
    parser.add_argument('--pop_size', type=int, default=100000, help='Population size')
    parser.add_argument('--N', type=int, default=20, help='Lattice size')
    parser.add_argument('--seed', type=int, default=224434057, help='Random seed')
    parser.add_argument('--Tstart', type=float, default=1.92, help='Starting temperature')
    parser.add_argument('--Tend', type=float, default=0.1, help='Ending temperature')
    parser.add_argument('--Cv_factor', type=float, default=1.618, help='Cv_factor')
    parser.add_argument('--MCsteps', type=int, default=1000, help='Number of Monte Carlo steps')
    parser.add_argument('--num_temps', type=int, default=30, help='Number of annealing temperatures')
    parser.add_argument('--schedule', type=str, default="Cv_beta", help='Scheduling of temperatures')
    
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
    N = args.N
    seed = args.seed
    Tstart = args.Tstart
    Tstart = float(Tstart)
    Tend = float(Tend)
    J = read_couplings(f'C:/Users/acer/Desktop/stage M1/MLMC_optimization/Data/Alpha/Couplings/Configurations/SK_N{N}_S{seed}.txt', N)
    with open(f'C:/Users/acer/Desktop/stage M1/MLMC_optimization/Data/Alpha/Couplings/Results/SK_sol_N{N}_S{seed}.txt') as f:
        lines = f.readlines()
     
        energie = float(lines[0].replace("#", "").strip())
        spins = np.array(list(map(int, lines[1].split())))


    MCsteps = args.MCsteps

    #perform the simulated annealing
    temperatures, observ, elapsed_time = simulated_annealing( J, pop_size, MCsteps, N, Tstart, Tend, Observables, num_temps_determiner = num_temps_determiner, schedule=schedule)

    minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
    print(MCsteps, f"{len(temperatures)}", schedule, f"{minimum:.5f}", f'{observ.get_observable_history("mean_energy")[-1]:.5f}', elapsed_time)