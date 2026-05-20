
import os
import re
from pathlib import Path
import torch
import numpy as np
import sys
import argparse
import time
sys.path.append("../../../Code/Legacy/packages")
sys.path.append("../../../Code/Modern/optimization")

from simulated_annealing_fast_XORSAT import *
from quantum_annealing_XORSAT import *
from PT_XORSAT import *
from population_annealing_XORSAT import *
from greedy_XORSAT import *
from global_annealing_XORSAT import *
from monte_carlo import *
path = '../../../../MLMC_optimization_XORSAT/Data/Alpha/Couplings/Configurations/'

txt_files = [f for f in os.listdir(path) if f.endswith(".txt")]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





if __name__ == "__main__":
    #parse all the possible arguments
    parser = argparse.ArgumentParser(description='Simulated Annealing Fast')
    parser.add_argument('--pop_size', type=int, default=10000, help='Population size')
    parser.add_argument('--algorithm', type=str, default='Greedy', help='algorithm that should be used')
    parser.add_argument('--Tstart', type=float, default=1.92, help='Starting temperature')
    parser.add_argument('--Tend', type=float, default=0.1, help='Ending temperature')
    parser.add_argument('--Cv_factor', type=float, default=1.618, help='Cv_factor')
    parser.add_argument('--MCsteps', type=int, default=320, help='Number of Monte Carlo steps')
    parser.add_argument('--num_temps', type=int, default=10, help='Number of annealing temperatures')
    parser.add_argument('--schedule', type=str, default="Cv_beta", help='Scheduling of temperatures')
    parser.add_argument('--MLMCsteps', type=int, default=5, help='Number of Machine-Learning assisted steps') 
    parser.add_argument('--swap_interval', type=int, default=10)
    #specific parsers for training
    parser.add_argument('--num_epochs_start', type=int, default=40, help='Number of epochs for the first training')
    parser.add_argument('--num_epochs_retrain', type=int, default=1, help='Number of epochs for each temperature retraining')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    #specific parsers for quantum anealing
    parser.add_argument("--P",            type=int,   default=20,
                        help="Number of Trotter slices")
    parser.add_argument("--Gamma_start",  type=float, default=2.0)
    parser.add_argument("--Gamma_end",    type=float, default=0.01)
    parser.add_argument("--schedule_Gamma", type=str, default="linear")
    parser.add_argument("--device",       type=str,   default=device)
    #optimisation
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
    Tstart = args.Tstart
    for file in txt_files:
        re.compile(r"xorsat_couplings_K(\d+)_J0_seed(\d+)\.t$")
        K    = int(re.search(r"K(\d+)_", file).group(1))
        seed = int(re.search(r"seed(\d+)\.", file).group(1))
        N=K*2
        J,h = read_couplings(path+file, N)
        J.to(device)
        h.to(device)

        print(file)

        Tstart = float(Tstart)
        Tend = float(Tend)
    
        MLMCsteps = args.MLMCsteps
        
        num_epochs_start = args.num_epochs_start
        num_epochs_retrain = args.num_epochs_retrain
        batch_size = args.batch_size
        MCsteps=args.MCsteps
        P=args.P
        algorithm=args.algorithm
        if algorithm=='Greedy':
            #Greedy
            observ, elapsed_time =greedy(K, J,h, pop_size,N,  Observables)
    
    
        elif algorithm=='SA':
            #SA
            temperatures, observ, elapsed_time = simulated_annealing(
                K, J,h,
                pop_size             = args.pop_size,
                num_steps_MC         = args.MCsteps,
                N                    = N,
                Tstart               = float(args.Tstart),
                Tend                 = float(args.Tend),
                Observables          = Observables,
                schedule             = args.schedule,
                num_temps_determiner = num_temps_determiner,
                adaptive_steps       = args.adaptive_steps,
                early_stop_tol       = args.early_stop_tol,
                thinning             = args.thinning,
                device               = args.device,)
    
    
        elif algorithm=='PT':
            #PT
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
    
        elif algorithm=='GA':     
            #GA
            temperatures, observ, elapsed_time = global_annealing(K, J,h, pop_size, MLMCsteps,
                                                                  MCsteps,N,  Tstart,
                                                                  Tend,  Observables, schedule = schedule,                                                                                                                                                             
                                                                  num_epochs_start = num_epochs_start,
                                                                  num_epochs_retrain = num_epochs_retrain,
                                                                  batch_size = batch_size,
                                                                  num_temps_determiner = num_temps_determiner)


        elif algorithm=='PA':     
            #PA
            temperatures, observ, elapsed_time = population_annealing(K, J,h, pop_size, MCsteps, 
                                                                      Tstart, Tend, Observables, 
                                                                      num_temps_determiner = num_temps_determiner, 
                                                                      schedule=schedule)

     
                                                                                 
     
        elif algorithm=='QA':
            #QA
            temperatures, observ, elapsed_time, best_configs = quantum_annealing_ST(
                K=K, J=J,h=h,
                pop_size=args.pop_size,
                P=args.P,
                num_steps_MC=args.MCsteps,
                Tstart=args.Tstart,
                Tend=args.Tend,
                Gamma_start=args.Gamma_start,
                Gamma_end=args.Gamma_end,
                Observables=Observables,
                schedule_T=args.schedule,
                schedule_Gamma=args.schedule_Gamma,
                num_temps=args.num_temps,
                device=args.device)
    
    
    
        else:
            print('Your program was not found')
        min_energy=torch.tensor(observ.get_observable_history("min_energy")).min()
        mean_energy=observ.get_observable_history('mean_energy')[-1]
        energy=torch.tensor(observ.get_observable_history("min_energy"))
        last_energy=observ.get_observable_history("min_energy")
        RESULTS_DIR = Path(__file__).resolve().parent / "../../../../MLMC_optimization_XORSAT/Data/Alpha/Couplings/Results"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(RESULTS_DIR / f"results_XORSAT_algo_{algorithm}_K{K}_seed{seed}.txt", "w") as f:
            f.write(f"K {K}\n")
            f.write(f"seed {seed}\n")
            f.write(f"minimum energy: {min_energy}\n")
            f.write(f"mean energy: {mean_energy}\n")
            f.write(f"time elaspsed: {elapsed_time}\n")
            f.write(" "+str(energy.tolist()) + "\n")
        print("SAVED: ")
        print(f'results_XORSAT_algo_{algorithm}_K{K}_seed{seed}.txt')
    
    
    
    
