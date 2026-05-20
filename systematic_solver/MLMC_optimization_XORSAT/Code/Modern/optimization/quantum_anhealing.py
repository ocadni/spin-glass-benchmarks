import numpy as np
import time
import torch
sys.path.append("C:/Users/acer/Desktop/stage M1/MLMC_optimization/Code/Legacy/packages")


def compute_energy_quantum(s, J,Gamma ,take_mean=False):
    """
    Compute the energy for each configuration using the Edwards-Anderson (EA) energy formula.

    Parameters:
    - s (torch.Tensor): Tensor representing the spin configurations. It is a #configurations x #numberofspin tensor.
    - J (torch.Tensor): Tensor representing the interaction matrix. It is a #numberofspin x #numberofspin tensor.
    - take_mean (bool): If True, compute the mean energy across all configurations. If False, return energies for each configuration.

    Returns:
    - torch.Tensor: Tensor containing the computed energy for each configuration or the mean energy.

    The function calculates the EA energy for each configuration in tensor 's' using
    the interaction matrix 'J'. It involves tensor contractions and multiplication
    operations. The computed energy values are returned as a tensor. If 'take_mean' is
    True, it returns the mean energy across all configurations.
    """
    # Perform tensor operations to calculate the EA energy for each configuration
    energy = -(torch.einsum("ki,ik->k", s, torch.einsum("ij,kj->ik", J, s))) / 2+Gamma * torch.einsum("ki->k", s)
    if not take_mean:
        return energy  # Return the computed energy tensor for each configuration
    else:
        return energy.mean()  # Return the mean energy across all configurations


def monte_carlo_update_fast_quantum(pop, J,beta, Gamma, even_indices, odd_indices):
    """Monte Carlo update model using a checkerboard pattern."""
    population = pop.clone()
    pop_size, N = population.shape
    # Define "even" and "odd" indices for a checkerboard update

    # Update spins in two passes (checkerboard pattern)
    for indices in [even_indices, odd_indices]:
        # Propose flips for the entire population at selected indices
        proposed_population = population.clone()
        proposed_population[:, indices] *= -1
        
        # Compute energy difference for each single-spin flip
        delta_E = -2* torch.einsum("ki, ki->ki",proposed_population[:, indices], torch.einsum("kj, ji->ki", population, J[indices, :].T))-2*Gamma * torch.einsum("ki->k", proposed_population[:, indices])
        
        # Metropolis acceptance criterion for each spin
        acceptance_prob = torch.exp(-beta * delta_E)
        random_vals = torch.rand(pop_size, len(indices), device=population.device)
        accept = (delta_E < 0) | (random_vals < acceptance_prob)
        
        # Apply accepted flips only for accepted positions
        population[:, indices] = torch.where(accept, proposed_population[:, indices], population[:, indices])

    return population


def quantum_annealing(L, J, pop_size, num_steps_MC, N, Tstart, Tend, Observables,Gamma_start, Gamma_end,
                                schedule = "Cv_beta", num_temps_determiner = 0.5, 
                                high_temp_thermalization_steps = 200, dimension = "3d",
                                schedule_Gamma="linear"):


    # ----- Annealing schedule for Gamma -----
    if schedule_Gamma == "linear":
        Gammas = torch.linspace(Gamma_start, Gamma_end, num_steps)
    elif schedule_Gamma == "exp":
        Gammas = Gamma_start * (Gamma_end / Gamma_start) ** (torch.linspace(0, 1, num_steps))
    else:
        raise ValueError("Unknown schedule")
    #get the indices (needed for the checkerboard update)
    if dimension == "3d":
        even_indices, odd_indices = get_indices(L)
        N = L*L*L
    elif dimension == "2d":
        even_indices, odd_indices = get_indices_2D(L)
        N = L*L
    else:
        raise ValueError("dimension must be either 3d or 2d")

    #set the temperature schedule
    temperatures = schedule_temperatures(Tstart, Tend, num_steps, schedule, N)

    #initialize the population
    population = torch.randint(0, 2, (pop_size,N), device="cuda").float() * 2 - 1
    
    #initialize the observables
    observ = Observables(J, N)

    # Thermalize the high temperature population
    for i in range(high_temp_thermalization_steps):
        population = monte_carlo_update_fast_quantum(population, J, beta=1/temperatures[0], Gamma=Gammas[0], even_indices=even_indices, odd_indices=odd_indices)
    observ.update(population) #save the minimum and mean energies


    start_time = time.time()


    for t in range(num_steps):
        Gamma = Gammas[t]
        T=temperatures[t]
        for i in range(num_steps_MC):
            population = monte_carlo_update_fast_quantum(population, J, beta=1/T, even_indices=even_indices, odd_indices=odd_indices)
        observ.update(population)   

    elapsed_time = time.time() - start_time

    return temperatures, observ, elapsed_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Tempering')

    parser.add_argument('--L', type=int, default=10)
    parser.add_argument('--seed', type=int, default=310411727)
    parser.add_argument('--Tstart', type=float, default=1.92)
    parser.add_argument('--Tend', type=float, default=0.1)
    parser.add_argument('--MCsteps', type=int, default=1000)
    parser.add_argument('--swap_interval', type=int, default=10)
    parser.add_argument('--num_temps', type=int, default=30)
    parser.add_argument('--schedule', type=str, default="linearT")
    parser.add_argument('--Gamma_start', type=float, default=2.0,
                    help='Initial transverse field (quantum fluctuations strength)')
    parser.add_argument('--Gamma_end', type=float, default=0.01,
                    help='Final transverse field (should be close to 0)')

    args = parser.parse_args()

    L = args.L
    N = L * L * L
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    J = read_couplings(f'C:/Users/acer/Desktop/stage M1/MLMC_optimization/Data/Alpha/Couplings/couplings_L{L}_R1_seed{seed}.txt', N)


    if args.schedule == "Cv_beta":
        num_temps_determiner = 1.618
    else:
        num_temps_determiner = args.num_temps

    temperatures, observ, elapsed_time = parallel_tempering(
        L, J,
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