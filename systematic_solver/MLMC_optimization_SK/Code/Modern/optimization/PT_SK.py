import torch
import numpy as np
import sys
import argparse
import time

sys.path.append("C:/Users/acer/Desktop/stage_M1/MLMC_optimization_SK/Code/Legacy/packages")

from geometry import *
from utilities import *
from data_loads import *
from monte_carlo import *


def parallel_tempering(N, J, num_replicas, num_steps_MC, swap_interval,
                      Tstart, Tend, Observables,
                      schedule="Cv_beta", num_temps_determiner=0.5,
                      high_temp_thermalization_steps=200,
                      dimension="3d"):



    # ----- Temperatures (one per replica) -----
    temperatures = schedule_temperatures(Tstart, Tend, num_temps_determiner, schedule, N)
    num_replicas = len(temperatures)

    betas = 1.0 / torch.tensor(temperatures)

    # ----- Initialize replicas -----
    replicas = torch.randint(0, 2, (num_replicas, N)).float() * 2 - 1

    # ----- Observables (track lowest-T replica) -----
    observ = Observables(J, N)

    # ----- Thermalize each replica -----
    for r in range(num_replicas):
        for _ in range(high_temp_thermalization_steps):
            replicas[r:r+1] = monte_carlo_update_fast(
                replicas[r:r+1], J,
                beta=betas[r],
            )

    observ.update(replicas[-1:])  # lowest temperature

    start_time = time.time()

    # ----- Main loop -----
    for step in range(num_steps_MC):

        # --- Monte Carlo updates ---
        for r in range(num_replicas):
            replicas[r:r+1] = monte_carlo_update_fast(
                replicas[r:r+1], J,
                beta=betas[r])

        # --- Swap step ---
        if step % swap_interval == 0:
            for r in range(num_replicas - 1):
                i, j = r, r + 1

                Ei = compute_energy(replicas[i:i+1], J)[0]
                Ej = compute_energy(replicas[j:j+1], J)[0]

                beta_i = betas[i]
                beta_j = betas[j]

                delta = (beta_j - beta_i) * (Ej - Ei)

                if torch.log(torch.rand(1)) < delta:
                    # swap configurations
                    tmp = replicas[i].clone()
                    replicas[i] = replicas[j]
                    replicas[j] = tmp

        # track lowest-T replica
        observ.update(replicas[-1:])

    end_time = time.time()
    elapsed_time = end_time - start_time

    return temperatures, observ, elapsed_time




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Tempering')

    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--Tstart', type=float, default=1.92)
    parser.add_argument('--Tend', type=float, default=0.1)
    parser.add_argument('--MCsteps', type=int, default=1000)
    parser.add_argument('--swap_interval', type=int, default=10)
    parser.add_argument('--num_temps', type=int, default=20)
    parser.add_argument('--schedule', type=str, default="linearT")

    args = parser.parse_args() 
    N=args.N 
    seed=args.seed
    J = read_couplings(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_SK/Data/Alpha/Couplings/Configurations/SK_N{N}_S{seed}.txt', N)
    with open(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_SK/Data/Alpha/Couplings/Results/SK_sol_N{N}_S{seed}.txt') as f:
        lines = f.readlines()
     
        energie = float(lines[0].replace("#", "").strip())
        spins = np.array(list(map(int, lines[1].split())))


    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)


    if args.schedule == "Cv_beta":
        num_temps_determiner = 1.618
    else:
        num_temps_determiner = args.num_temps

    temperatures, observ, elapsed_time = parallel_tempering(
        N, J,
        num_replicas=args.num_temps,
        num_steps_MC=args.MCsteps,
        swap_interval=args.swap_interval,
        Tstart=args.Tstart,
        Tend=args.Tend,
        Observables=Observables,
        schedule=args.schedule,
        num_temps_determiner=num_temps_determiner
    )

    minimum = torch.tensor(observ.get_observable_history("min_energy")).min()

    print(args.MCsteps,
          len(temperatures),
          args.schedule,
          f"{minimum:.5f}",
          f"{observ.get_observable_history('mean_energy')[-1]:.5f}",
          elapsed_time)