import torch
import numpy as np
import sys
import argparse
import time

sys.path.append("C:/Users/acer/Desktop/stage M1/MLMC_optimization/Code/Legacy/packages")

from geometry import *
from utilities import *
from data_loads import *
from monte_carlo import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parallel_tempering(L, J,h, num_replicas,P, num_steps_MC, swap_interval,
                      Tstart, Tend, Observables,
                      schedule="Cv_beta", num_temps_determiner=0.5,
                      high_temp_thermalization_steps=200):

    N=K*2
    INDICES=greedy_independent_sets(J)[0]
    # ----- Temperatures (one per replica) -----
    temperatures = schedule_temperatures(Tstart, Tend, num_temps_determiner, schedule, N)
    num_replicas = len(temperatures)

    betas = 1.0 / torch.tensor(temperatures)

    # ----- Initialize replicas -----
    replicas = (torch.randint(0, 2, (P,num_replicas, N), device=device).float() * 2 - 1)

    # ----- Observables (track lowest-T replica) -----
    observ = Observables(J, N,h)

    # ----- Thermalize each replica -----
    for r in range(num_replicas):
        for _ in range(high_temp_thermalization_steps):
            replicas[r:r+1] = monte_carlo_update_fast(
                replicas[r:r+1], J,h,
                betas[r],INDICES)

    observ.update(replicas[-1:])  # lowest temperature

    start_time = time.time()

    # ----- Main loop -----
    for step in range(num_steps_MC):

        # --- Monte Carlo updates ---
        for r in range(num_replicas):
            replicas[r:r+1] = monte_carlo_update_fast(
                replicas[r:r+1], J,h,
                betas[r],
                INDICES
            )

        # --- Swap step ---
        if step % swap_interval == 0:
            for r in range(num_replicas - 1):
                i, j = r, r + 1

                Ei = compute_energy(replicas[i:i+1], J,h)[0]
                Ej = compute_energy(replicas[j:j+1], J,h)[0]

                beta_i = betas[i]
                beta_j = betas[j]

                delta = (beta_i - beta_j) * (Ej - Ei)

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

    parser.add_argument('--K', type=int, default=99)
    parser.add_argument('--seed', type=int, default=345692)
    parser.add_argument('--Tstart', type=float, default=1.92)
    parser.add_argument('--Tend', type=float, default=0.1)
    parser.add_argument('--MCsteps', type=int, default=1000)
    parser.add_argument('--swap_interval', type=int, default=10)
    parser.add_argument('--num_temps', type=int, default=30)
    parser.add_argument('--schedule', type=str, default="linearT")

    args = parser.parse_args()

    K = args.K
    N = K*2
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    J = read_couplings(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Data/Alpha/Couplings/couplings_XORSAT_K{K}_seed{seed}.txt', N).to(device)
    h=read_field(f'C:/Users/acer/Desktop/stage_M1/MLMC_optimization_XORSAT/Data/Alpha/Couplings/field_XORSAT_K{K}_seed{seed}.txt', N).to(device)
    

    if args.schedule == "Cv_beta":
        num_temps_determiner = 1.618
    else:
        num_temps_determiner = args.num_temps

    temperatures, observ, elapsed_time = parallel_tempering(
        K, J,h,
        num_replicas=args.num_temps,
        num_steps_MC=args.MCsteps,
        swap_interval=args.swap_interval,
        Tstart=args.Tstart,
        Tend=args.Tend,
        Observables=Observables,
        schedule=args.schedule,
        num_temps_determiner=num_temps_determiner)
    minimum = torch.tensor(observ.get_observable_history("min_energy")).min() 
    real_minimum = torch.tensor(observ.get_observable_history("min_real_energy")).min()
    print( f"{minimum:.5f}", f"{observ.get_observable_history('mean_energy')[-1]:.5f}", elapsed_time)
    print( f"{real_minimum:.5f}", f"{observ.get_observable_history('mean_real_energy')[-1]:.5f}", elapsed_time)
 
    print(args.MCsteps,
          len(temperatures),
          args.schedule,
          f"{minimum:.5f}",
          f"{observ.get_observable_history('mean_energy')[-1]:.5f}",
          elapsed_time)