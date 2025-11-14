import numpy as np
import math

def simulated_annealing(instance, temp_initial=10.0, temp_final=0.01, cooling_rate=0.99, steps_per_temp=100):
    """
    Minimal Simulated Annealing solver.

    Args:
        instance (tuple): (N, couplings) from a generator.
        temp_initial (float): Initial temperature.
        temp_final (float): Final temperature.
        cooling_rate (float): Cooling rate (geometric).
        steps_per_temp (int): Number of Monte Carlo steps at each temperature.

    Returns:
        tuple: (best_energy, best_state)
    """
    N, couplings = instance
    
    # Build adjacency list for faster energy calculation
    adj = {i: [] for i in range(N)}
    for i, j, J in couplings:
        adj[i].append((j, J))
        adj[j].append((i, J))

    def calculate_total_energy(state):
        energy = 0
        for i in range(N):
            for j, J in adj[i]:
                if i < j:
                    energy -= J * state[i] * state[j]
        return energy

    # 1. Initialization
    current_state = np.random.choice([-1, 1], size=N)
    current_energy = calculate_total_energy(current_state)
    
    best_state = current_state.copy()
    best_energy = current_energy
    
    temp = temp_initial

    # 2. Cooling loop
    while temp > temp_final:
        for _ in range(steps_per_temp):
            # 3. Propose a new state (flip one spin)
            spin_to_flip = np.random.randint(N)
            
            # 4. Calculate energy change
            delta_E = 0
            for neighbor, J in adj[spin_to_flip]:
                delta_E += 2 * J * current_state[spin_to_flip] * current_state[neighbor]

            # 5. Acceptance criterion
            if delta_E < 0 or np.random.rand() < math.exp(-delta_E / temp):
                current_state[spin_to_flip] *= -1
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = current_state.copy()
        
        # 6. Cool down
        temp *= cooling_rate
        
    return best_energy, best_state

if __name__ == '__main__':
    from generators.generate_sk import generate_sk
    
    instance = generate_sk(N=30)
    energy, state = simulated_annealing(instance)
    
    print(f"SA finished.")
    print(f"Best energy: {energy}")
