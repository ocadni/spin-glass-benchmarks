import numpy as np

def population_annealing(instance, num_replicas=100, beta_min=0.01, beta_max=5.0, num_beta_steps=100):
    """
    Minimal Population Annealing solver.

    Args:
        instance (tuple): (N, couplings) from a generator.
        num_replicas (int): Number of replicas in the population.
        beta_min (float): Starting inverse temperature.
        beta_max (float): Final inverse temperature.
        num_beta_steps (int): Number of temperature steps.

    Returns:
        tuple: (best_energy, best_state)
    """
    N, couplings = instance
    
    # Build adjacency list for faster energy calculation
    adj = {i: [] for i in range(N)}
    for i, j, J in couplings:
        adj[i].append((j, J))
        adj[j].append((i, J))

    def calculate_energy(state):
        energy = 0
        for i in range(N):
            for j, J in adj[i]:
                if i < j:
                    energy -= J * state[i] * state[j]
        return energy

    # 1. Initialization
    population = np.random.choice([-1, 1], size=(num_replicas, N))
    betas = np.linspace(beta_min, beta_max, num_beta_steps)

    best_energy = float('inf')
    best_state = None

    # 2. Main PA loop
    for beta in betas:
        # 2a. Gibbs sampling (or Metropolis updates) for each replica
        for r in range(num_replicas):
            for _ in range(N): # One sweep
                spin_to_flip = np.random.randint(N)
                delta_E = 0
                for neighbor, J in adj[spin_to_flip]:
                    delta_E += 2 * J * population[r, spin_to_flip] * population[r, neighbor]
                
                if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
                    population[r, spin_to_flip] *= -1
        
        # 2b. Resampling
        energies = np.array([calculate_energy(s) for s in population])
        weights = np.exp(-beta * (energies - np.min(energies)))
        weights /= np.sum(weights)
        
        resampling_indices = np.random.choice(num_replicas, size=num_replicas, p=weights)
        population = population[resampling_indices]

        # Update best found solution
        min_idx = np.argmin(energies)
        if energies[min_idx] < best_energy:
            best_energy = energies[min_idx]
            best_state = population[min_idx].copy()

    return best_energy, best_state


if __name__ == '__main__':
    from generators.generate_sk import generate_sk
    
    instance = generate_sk(N=20)
    energy, state = population_annealing(instance)
    
    print(f"PA finished.")
    print(f"Best energy: {energy}")
