import numpy as np
import itertools

def generate_ea(L, dim=2, j_dist='bimodal'):
    """
    Generates a 2D or 3D Edwards-Anderson (EA) model instance.

    Args:
        L (int): The linear size of the lattice.
        dim (int): The dimension of the lattice (2 or 3).
        j_dist (str): The distribution of coupling strengths ('bimodal' or 'gaussian').

    Returns:
        tuple: A tuple containing:
            - N (int): The number of spins.
            - couplings (list): A list of (i, j, J_ij) tuples.
    """
    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3.")

    N = L ** dim
    couplings = []
    
    if j_dist == 'bimodal':
        J = np.random.choice([-1.0, 1.0], size=N*dim)
    elif j_dist == 'gaussian':
        J = np.random.normal(0, 1, size=N*dim)
    else:
        raise ValueError("j_dist must be 'bimodal' or 'gaussian'.")

    j_idx = 0
    for i in range(N):
        coords = []
        temp_i = i
        for d in range(dim - 1, -1, -1):
            coord = temp_i // (L ** d)
            coords.append(coord)
            temp_i %= (L ** d)
        
        for d in range(dim):
            # Periodic boundary conditions
            neighbor_coords = list(coords)
            neighbor_coords[d] = (neighbor_coords[d] + 1) % L
            
            j = 0
            for k in range(dim):
                j += neighbor_coords[k] * (L ** (dim - 1 - k))

            if i < j:
                couplings.append((i, j, J[j_idx]))
                j_idx += 1

    return N, couplings

if __name__ == '__main__':
    N, couplings = generate_ea(L=4, dim=2)
    print(f"Generated 2D EA model with {N} spins.")
    # for c in couplings:
    #     print(f"{c[0]} {c[1]} {c[2]}")
