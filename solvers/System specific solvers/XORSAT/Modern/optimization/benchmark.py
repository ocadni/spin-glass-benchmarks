import torch
import numpy as np
import sys
import argparse
import time
sys.path.append("../../../Code/Legacy/packages")
sys.path.append("../../../Code/Modern/optimization")

from simulated_annealing_fast import *
from quantum_annealing_gpu import *
from PT import *
from population_annealing import *
from greedy import *
from global_annealing import *
#C:\Users\acer\Desktop\stage M1\MLMC_optimization\Code\Modern\optimization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

freq=30
MC_min=1
MC_max=5000
N=10
arr_MC = np.logspace(np.log10(MC_min), np.log10(MC_max), N)
SA=np.zeros(N)
PT=np.zeros(N)
GA=np.zeros(N)
QA=np.zeros(N)
Greedy=np.zeros(N)
PA=np.zeros(N)
time_SA=np.zeros((N,freq))
time_PT=np.zeros((N,freq))
time_GA=np.zeros((N,freq))
time_QA=np.zeros((N,freq))
time_Greedy=np.zeros((N,freq))
time_PA=np.zeros((N,freq))
min_SA=np.zeros((N,freq))
min_PT=np.zeros((N,freq))
min_GA=np.zeros((N,freq))
min_QA=np.zeros((N,freq))
min_Greedy=np.zeros((N,freq))
min_PA=np.zeros((N,freq))



if __name__ == "__main__":
    #parse all the possible arguments
    parser = argparse.ArgumentParser(description='Simulated Annealing Fast')
    parser.add_argument('--pop_size', type=int, default=100000, help='Population size')
    parser.add_argument('--L', type=int, default=12, help='Lattice size')
    parser.add_argument('--seed', type=int, default=310411727, help='Random seed')
    parser.add_argument('--Tstart', type=float, default=1.92, help='Starting temperature')
    parser.add_argument('--Tend', type=float, default=0.1, help='Ending temperature')
    parser.add_argument('--Cv_factor', type=float, default=1.618, help='Cv_factor')
    parser.add_argument('--MCsteps', type=int, default=320, help='Number of Monte Carlo steps')
    parser.add_argument('--num_temps', type=int, default=40, help='Number of annealing temperatures')
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
    L = args.L
    N = L*L*L
    seed = args.seed
    Tstart = args.Tstart
    J = read_couplings(f'/home/landacunha/dossier_project/stage M1/MLMC_optimization/Data/Alpha/Couplings/couplings_L{L}_R1_seed{seed}.txt', N).to(device)

    Tstart = float(Tstart)
    Tend = float(Tend)

    MLMCsteps = args.MLMCsteps
    
    num_epochs_start = args.num_epochs_start
    num_epochs_retrain = args.num_epochs_retrain
    batch_size = args.batch_size
    
    P=args.P







true_min=-1.69287
for i in range (N):
    MCsteps = int(arr_MC[i])
    for k in range(freq):
        #Greedy
        observ, elapsed_time = greedy(L, J, pop_size,N,  Observables)
        minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
        time_Greedy[i][k]=elapsed_time
        min_Greedy[i][k]=minimum
np.savetxt("time_Greedy", arr, fmt="%.10f", delimiter=",")
np.savetxt("min_Greedy", arr, fmt="%.10f", delimiter=",")


for i in range (N):
    MCsteps = int(arr_MC[i])
    for k in range(freq):
        #SA
        temperatures, observ, elapsed_time = simulated_annealing(L, J, pop_size, MCsteps, N, Tstart, Tend, Observables, num_temps_determiner = num_temps_determiner, schedule=schedule)
        minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
        time_SA[i][k]=elapsed_time
        min_SA[i][k]=minimum
np.savetxt("time_SA", arr, fmt="%.10f", delimiter=",")
np.savetxt("min_SA", arr, fmt="%.10f", delimiter=",")


for i in range (N):
    MCsteps = int(arr_MC[i])
    for k in range(freq):
        #PT
        temperatures, observ, elapsed_time = parallel_tempering(
            L, J,
            num_replicas=args.num_temps,
            num_steps_MC=MCsteps,
            swap_interval=args.swap_interval,
            Tstart=args.Tstart,
            Tend=args.Tend,
            Observables=Observables,
            schedule=args.schedule,
            num_temps_determiner=num_temps_determiner)
        minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
        time_PT[i][k]=elapsed_time
        min_PT[i][k]=minimum
np.savetxt("time_PT", arr, fmt="%.10f", delimiter=",")
np.savetxt("min_PT", arr, fmt="%.10f", delimiter=",")

for i in range (N):
    MCsteps = int(arr_MC[i])
    for k in range(freq):        
        #GA
        temperatures, observ, elapsed_time = global_annealing(L, J, pop_size, MLMCsteps, MCsteps,N,  Tstart,
                                                                                           Tend,  Observables, schedule = schedule,                                                                                                                                                             
                                                                                           num_epochs_start = num_epochs_start,
                                                                                           num_epochs_retrain = num_epochs_retrain,
                                                                                           batch_size = batch_size,
                                                                                           num_temps_determiner = num_temps_determiner)
        minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
        time_GA[i][k]=elapsed_time
        min_GA[i][k]=minimum
np.savetxt("time_GA", arr, fmt="%.10f", delimiter=",")
np.savetxt("min_GA", arr, fmt="%.10f", delimiter=",")
 

pop_size = args.pop_size//P 
for i in range (N):
    MCsteps = int(arr_MC[i])
    for k in range(freq):      
        #QA
        temperatures, observ, elapsed_time, best_configs = quantum_annealing_ST(
            L=L, J=J,
            pop_size=pop_size,
            P=args.P,
            num_steps_MC=MCsteps,
            Tstart=args.Tstart,
            Tend=args.Tend,
            Gamma_start=args.Gamma_start,
            Gamma_end=args.Gamma_end,
            Observables=Observables,
            schedule_T=args.schedule,
            schedule_Gamma=args.schedule_Gamma,
            num_temps=args.num_temps,
            device=args.device,)
        minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
        time_QA[i][k]=elapsed_time
        min_QA[i][k]=minimum
pop_size = args.pop_size
np.savetxt("time_QA", arr, fmt="%.10f", delimiter=",")
np.savetxt("min_QA", arr, fmt="%.10f", delimiter=",")


for i in range (N):
    MCsteps = int(arr_MC[i])
    for k in range(freq):
        #PA
        temperatures, observ, elapsed_time = population_annealing(L, J, pop_size, MCsteps, Tstart, Tend, Observables, num_temps_determiner = num_temps_determiner, schedule=schedule)
        minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
        time_PA[i][k]=elapsed_time
        min_PA[i][k]=minimum
np.savetxt("time_PA", arr, fmt="%.10f", delimiter=",")
np.savetxt("min_PA", arr, fmt="%.10f", delimiter=",")





SA=SA/freq
PT=PT/freq
GA=GA/freq
QA=QA/freq
Greedy=Greedy/freq
PA=PA/freq
