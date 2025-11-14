import numpy as np
import networkx as nx

def generate_rrg(N, k, j_dist='bimodal'):
    """
    Generates a Random Regular Graph (RRG) model instance.

    Args:
        N (int): The number of spins.
        k (int): The degree of each node.
        j_dist (str): The distribution of coupling strengths ('bimodal' or 'gaussian').

    Returns:
        tuple: A tuple containing:
            - N (int): The number of spins.
            - couplings (list): A list of (i, j, J_ij) tuples.
    """
    if (N * k) % 2 != 0:
        raise ValueError("N * k must be even.")

    graph = nx.random_regular_graph(k, N)
    num_edges = graph.number_of_edges()

    if j_dist == 'bimodal':
        J = np.random.choice([-1.0, 1.0], size=num_edges)
    elif j_dist == 'gaussian':
        J = np.random.normal(0, 1, size=num_edges)
    else:
        raise ValueError("j_dist must be 'bimodal' or 'gaussian'.")

    couplings = []
    for idx, (i, j) in enumerate(graph.edges()):
        couplings.append((i, j, J[idx]))

    return N, couplings

if __name__ == '__main__':
    N, couplings = generate_rrg(N=20, k=3)
    print(f"Generated RRG model with {N} spins and degree 3.")
    # for c in couplings:
    #     print(f"{c[0]} {c[1]} {c[2]}")
