import numpy as np

def generate_sk(N, j_dist='gaussian'):
    """
    Generates a Sherrington-Kirkpatrick (SK) model instance.

    Args:
        N (int): The number of spins.
        j_dist (str): The distribution of coupling strengths ('bimodal' or 'gaussian').

    Returns:
        tuple: A tuple containing:
            - N (int): The number of spins.
            - couplings (list): A list of (i, j, J_ij) tuples.
    """
    num_couplings = N * (N - 1) // 2
    
    if j_dist == 'bimodal':
        J = np.random.choice([-1.0, 1.0], size=num_couplings)
    elif j_dist == 'gaussian':
        # Standard SK model has variance 1/N
        J = np.random.normal(0, 1.0 / np.sqrt(N), size=num_couplings)
    else:
        raise ValueError("j_dist must be 'bimodal' or 'gaussian'.")

    couplings = []
    j_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            couplings.append((i, j, J[j_idx]))
            j_idx += 1
            
    return N, couplings

if __name__ == '__main__':
    N, couplings = generate_sk(N=10)
    print(f"Generated SK model with {N} spins.")
    # for c in couplings:
    #     print(f"{c[0]} {c[1]} {c[2]}")
