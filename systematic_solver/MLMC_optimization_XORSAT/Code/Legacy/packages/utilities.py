"""Utilities functions. Stuff than can be used in general for parallel tempering and often comes in handy"""

#imports

#general imports
import numpy as np
import torch

#C_v definition and functions that generate temp_range

def Cv(T, N):
    """Approximation of heat capacity C_V by FRT.

    Parameters:
    - T: temperature,
    - N: number of spins.

    Returns:
    - C_V: heat capacity"""
    g = 0.05 * T + 2.0 * T * T * T
    return N * g / (1.0 + (g * 0.5 * T * T)**0.4)**2.5

def Cv_3d(T, N):
    """Approximation of heat capacity C_V by FRT for 3D Ising model.

    Parameters:
    - T: temperature,
    - N: number of spins.
    
    Returns:
    - C_V: heat capacity"""
    return N * 34.19 * T * T / (10.36 + T * T * T) / (10.36 + T * T * T)

def get_temps(start, finish, N):
    """
    Generate temperatures using the heat capacity policy.

    Parameters:
    - start (float): Starting temperature.
    - finish (float): Target temperature to stop the generation.
    - N (int): Number of spins

    Returns:
    - numpy.ndarray: Array of generated temperatures.

    The function generates a sequence of temperatures starting from 'start' and
    continuing until reaching or exceeding 'finish' using a specific policy. The
    policy involves modifying the temperature 'T' based on the optimal thermal step of the Parallel
    Tempering procedure.
    """
    T = start  # Initialize the temperature with the starting value
    temperatures = [T]  # Create a list to store generated temperatures, starting with the initial temperature

    # Continue generating temperatures until the temperature reaches or exceeds the 'finish' value
    while T < finish:
        # Update the temperature using a specific formula (commented out line can be an alternative)
        T /= (1.0 - 1.684 / np.sqrt(Cv(T, N)))
        # Uncomment the line below if the approximate version is used
        # T *= 1.0 + 1.684 / np.sqrt(Cv(T, N))

        temperatures.append(T)  # Append the updated temperature to the list

    return np.array(temperatures)  # Convert the list to a numpy array and return it

def get_betas(start, finish, factor, N):
    """
    Generate beta values using a specific factor as a multiplying factor of 1/sqrt(Cv).

    Parameters:
    - start: Starting beta value.
    - finish: Target beta value to stop the generation.
    - factor: Multiplying factor for 1/sqrt(Cv) in the generation process.
    - N: Number of spins
    Returns:
    -betas: Array of generated beta values.

    The function generates a sequence of beta values starting from 'start' and
    continuing until reaching or exceeding 'finish'. The generation process involves
    modifying beta based on the PT formula. The
    generated beta values are stored in a numpy array and returned.
    """

    beta = start  # Initialize the beta value with the starting value
    betas = [beta]  # Create a list to store generated beta values, starting with the initial beta

    # Continue generating beta values until the beta value reaches or exceeds the 'finish' value
    while beta < finish:
        # Update beta using a specific formula
        beta += factor / np.sqrt(Cv(1 / beta, N))

        betas.append(beta)  # Append the updated beta value to the list

    return np.array(betas)  # Convert the list to a numpy array and return it

def get_betas_3d(start, finish, factor, N):
    """
    Generate beta values using a specific factor as a multiplying factor of 1/sqrt(Cv) for Ising 3d.

    Parameters:
    - start: Starting beta value.
    - finish: Target beta value to stop the generation.
    - factor: Multiplying factor for 1/sqrt(Cv) in the generation process.
    - N: Number of spins
    Returns:
    -betas: Array of generated beta values.

    The function generates a sequence of beta values starting from 'start' and
    continuing until reaching or exceeding 'finish'. The generation process involves
    modifying beta based on the PT formula. The
    generated beta values are stored in a numpy array and returned.
    """

    beta = start  # Initialize the beta value with the starting value
    betas = [beta]  # Create a list to store generated beta values, starting with the initial beta

    # Continue generating beta values until the beta value reaches or exceeds the 'finish' value
    while beta < finish:
        # Update beta using a specific formula
        beta += factor / np.sqrt(Cv_3d(1 / beta, N))

        betas.append(beta)  # Append the updated beta value to the list

    return np.array(betas)  # Convert the list to a numpy array and return it

def compute_energy(s, J,h, take_mean=False):
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
    energy = -(torch.einsum("ki,ik->k", s, torch.einsum("ij,kj->ik", J, s))) / 2
    energy -= torch.einsum("ki,i->k", s, h)
    if not take_mean:
        return energy  # Return the computed energy tensor for each configuration
    else:
        return energy.mean()  # Return the mean energy across all configurations
    
    
def compute_one_energy(s,J,h):
    energy = -(torch.sum( s*torch.matmul( J, s))) / 2
    energy -= torch.sum(s*h)
    return energy
    
def compute_one_real_energy(s,J,h):
    half=len(s)//2
    energy = -(torch.sum( s[:half]*torch.matmul( J[:half,:half], s[:half]))) / 2
    energy -= torch.sum(s[:half]*h[:half])
    return energy

def physical_energy(s,J,h):
    half=len(s)
    energy = -(torch.sum( s*torch.matmul( J[:half,:half], s))) / 2
    energy -= torch.sum(s*h[:half])
    return energy

def compute_real_energy(s, J, h, take_mean=False):
    """
    Compute EA energy using only first N/2 spins.
    """
    N = s.shape[1]
    half = N // 2

    # truncate spins
    s = s[:, :half]
    h = h[:half]
    J = J[:half, :half]

    energy = -(torch.einsum("ki,ik->k", s, torch.einsum("ij,kj->ik", J, s))) / 2
    energy -= torch.einsum("ki,i->k", s, h)

    if not take_mean:
        return energy
    else:
        return energy.mean()
    
def hamming_distance(data, GS_config):
    # Calculate the absolute differences between GS_config and each row in data
    distance1 = torch.sum(torch.abs(GS_config - data), axis=1)
    
    # Calculate the absolute differences between the negation of GS_config and each row in data
    distance2 = torch.sum(torch.abs(GS_config + data), axis=1)
    
    # Take the element-wise minimum of distance1 and distance2
    x = torch.min(distance1, distance2)
    
    # Calculate the mean of the minimum distances and divide by 2
    result = torch.mean(x) / 2
    
    return result

def corr(x, tau):
  """Compute autocorrelation function"""
  if tau == 0:
    return torch.mean(x*x)
  else:
    return torch.mean(x[tau:,:]*x[:-tau, :])-torch.mean(x[tau:,:])*torch.mean(x[:-tau,:])













def build_matrix(triples, b=None, N=None):
    """
    Build matrix A and vector b over GF(2)
    triples: list of (p,q,k)
    b: list/array of 0/1 (default = all zeros)
    N: number of variables (if None, inferred)
    """
    M = len(triples)

    if N is None:
        N = max(max(t) for t in triples) + 1

    A = np.zeros((M, N), dtype=np.uint8)

    for i, (p, q, k) in enumerate(triples):
        A[i, p] = 1
        A[i, q] = 1
        A[i, k] = 1

    if b is None:
        b = np.zeros(M, dtype=np.uint8)
    else:
        b = np.array(b, dtype=np.uint8)

    return A, b


def gaussian_elimination_mod2(A, b):
    """
    Perform Gaussian elimination over GF(2)
    Returns row-echelon form and pivot columns
    """
    A = A.copy()
    b = b.copy()

    M, N = A.shape
    row = 0
    pivots = []

    for col in range(N):
        pivot = None
        for r in range(row, M):
            if A[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            continue

        # swap rows
        A[[row, pivot]] = A[[pivot, row]]
        b[[row, pivot]] = b[[pivot, row]]

        # eliminate
        for r in range(M):
            if r != row and A[r, col] == 1:
                A[r] ^= A[row]
                b[r] ^= b[row]

        pivots.append(col)
        row += 1

        if row == M:
            break

    return A, b, pivots

"""
def extract_solution(A, b, pivots, N):
    #Extract:
    #- particular solution
    #- kernel basis
    #- dimension

    M = A.shape[0]
    pivot_set = set(pivots)

    free_vars = [j for j in range(N) if j not in pivot_set]
    d = len(free_vars)

    # particular solution (set free vars = 0)
    x0 = np.zeros(N, dtype=np.uint8)

    for i, col in enumerate(pivots):
        x0[col] = b[i]

    # kernel basis
    basis = []

    for free in free_vars:
        v = np.zeros(N, dtype=np.uint8)
        v[free] = 1

        for i, col in enumerate(pivots):
            if A[i, free] == 1:
                v[col] = 1

        basis.append(v)

    return x0, basis, d
"""
def extract_solution(A, b, pivots, N):
    pivot_set = set(pivots)
    free_vars = [j for j in range(N) if j not in pivot_set]

    x0 = np.zeros(N, dtype=np.uint8)

    # particular solution
    for i, col in enumerate(pivots):
        x0[col] = b[i] % 2

    basis = []

    for free in free_vars:
        v = np.zeros(N, dtype=np.uint8)
        v[free] = 1

        for i, col in enumerate(pivots):
            if A[i, free] == 1:
                v[col] = (v[col] + 1) % 2

        basis.append(v % 2)

    return x0 % 2, basis, len(free_vars)

def solve_xorsat(triples, b=None, N=None):
    """
    Full pipeline:
    returns:
        x0        -> particular solution (or None if no solution)
        basis     -> list of kernel vectors
        dim       -> dimension of solution space
    """
    A, b = build_matrix(triples, b, N)
    M, N = A.shape

    A_red, b_red, pivots = gaussian_elimination_mod2(A, b)

    # check consistency
    for i in range(M):
        if np.all(A_red[i] == 0) and b_red[i] == 1:
            return None, None, 0  # no solution

    x0, basis, dim = extract_solution(A_red, b_red, pivots, N)

    return x0, basis, dim

def solution(SET,b,J,h):
    x0, basis, dim=solve_xorsat(SET, b, N=None)
    # Propose flips for the entire population at selected indices
    K=len(x0)
    indices=range(K,K*2)
    population = torch.zeros(K*2)
    population[:K]+=x0
    population= population*2-1
    proposed_population = population.clone()
    proposed_population[indices] *= -1
    
    # Compute energy difference for each single-spin flip
    local_field = torch.einsum("j,ji->i", population, J[indices, :].T) + h[indices]
    delta_E = 2 * population[indices] * local_field
    
    # Metropolis acceptance criterion for each spin
    acceptance_prob =torch.sign(-delta_E)/2+0.5
    # if the local minimum is reached then no fliping will be accepted
    if torch.sum(acceptance_prob)==0:
        return population
    # Apply accepted flips only for accepted positions
    accept=acceptance_prob.bool()
    population[indices] = torch.where(accept, proposed_population[ indices], population[ indices])

    return population


def complete_solution(s,J,h):
    K=len(s)
    indices=range(K,K*2)
    population = torch.zeros(K*2)
    population[:K]+=s
    population[K:]+=torch.zeros(K)+1
    proposed_population = population.clone()
    proposed_population[indices] *= -1
    
    # Compute energy difference for each single-spin flip
    local_field = torch.einsum("j,ji->i", population, J[indices, :].T) + h[indices]
    delta_E = 2 * population[indices] * local_field
    
    # Metropolis acceptance criterion for each spin
    acceptance_prob =torch.sign(-delta_E)/2+0.5
    # if the local minimum is reached then no fliping will be accepted
    if torch.sum(acceptance_prob)==0:
        return population
    # Apply accepted flips only for accepted positions
    accept=acceptance_prob.bool()
    population[indices] = torch.where(accept, proposed_population[ indices], population[ indices])

    return population




def TTSp(energies,E_ground_state):
    sucess=np.abs(torch.stack(energies)-E_ground_state)<1e-5
    sucess.int()
    TTS=sucess.sum(axis=1)/sucess.shape[1]
    return TTS