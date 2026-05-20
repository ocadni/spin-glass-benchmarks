"""
generate_couplings_nn.py
Nearest-neighbor Ising couplings on a 2D or 3D periodic lattice.

Each spin is connected only to its 2d (d=2) or 2d (d=3) nearest neighbours.
The coupling matrix is symmetric: only pairs (p, q) with p < q are written,
each line containing  p  q  J(p,q)  where J ~ N(0, J_sigma).

"""
import torch
import argparse
import numpy as np
import random
import sys
from collections import Counter
from itertools import product
sys.path.append("../../../Code/Legacy/packages")
sys.path.append("../../../Code/Modern/optimization")

from utilities import *
from greedy_XORSAT import *

def generate_sets(N: int, max_attempts: int = 100_000) -> list[tuple[int, int, int]]:
    """
    Return N sorted 3-tuples satisfying all constraints.
 
    Parameters
    ----------
    N             : must be divisible by 3 and ≥ 3
    max_attempts  : safety limit on random restarts
 
    Raises
    ------
    ValueError   if N is invalid
    RuntimeError if no solution found within max_attempts
    """
    if not isinstance(N, int) or N < 6:
        raise ValueError("N must be an integer ≥ 7")
    if N % 6 != 1 and N%6 != 3:
        raise ValueError(f"K must be K=1[6] or K=3[6] (got {N})")
 
    for attempt in range(1, max_attempts + 1):
        result = _try_generate(N)
        if result is not None:
            return result
 
    raise RuntimeError(
        f"Could not generate a valid solution for N={N} "
        f"after {max_attempts} attempts."
    )
 
''' 
def _try_generate(N: int) -> list[tuple] | None:
    """Single attempt: shuffle three rows and zip into sets."""
    rows = [list(range(N)) for _ in range(3)]
    for row in rows:
        random.shuffle(row)
 
    sets: list[tuple] = []
    seen: set[tuple] = set()
 
    for a, b, c in zip(*rows):
        s = tuple(sorted({a, b, c}))
 
        if len(s) < 3:   # duplicate value within this set
            return None
        if s in seen:    # duplicate set
            return None
 
        seen.add(s)
        sets.append(s)
 
    return sets
'''  
def _try_generate(N: int):
    rows = [list(range(N)) for _ in range(3)]
    for row in rows:
        random.shuffle(row)

    sets = []
    seen_triples = set()
    used_pairs = set()   # 🔑 track used pairs

    for a, b, c in zip(*rows):
        triple = tuple(sorted((a, b, c)))  # ensures p < q < k

        if len(set(triple)) < 3:
            return None

        if triple in seen_triples:
            return None

        # 🔥 check pair reuse
        p, q, k = triple
        pairs = [(p, q), (p, k), (q, k)]

        for pair in pairs:
            if pair in used_pairs:
                return None  # reject this attempt

        # accept
        seen_triples.add(triple)
        sets.append(triple)
        used_pairs.update(pairs)

    return sets


def generate_XORSAT(K, rng):

    pairs = {}
    h = {}
    if K % 6 != 3 and K%6 != 1:
        print('L is not divisible by 3!')

    SET = generate_sets(K)
    
    # ── Plant a solution first ──────────────────────────────────────────
    # Draw a random physical spin config as the planted solution
    x_planted = torch.randint(0, 2, (K,)).float()  # xi in {0,1}
    s_planted  = 2 * x_planted -1                            # si in {+1,-1}
    # Derive b from the planted solution: b_i = (1 - s_p*s_q*s_k) / 2
    b = torch.zeros(K)
    for i in range(K):
        p, q, k = SET[i][0], SET[i][1], SET[i][2]
        product = s_planted[p] * s_planted[q] * s_planted[k]
        b[i] = (1 - product) / 2   # 0 if product=+1, 1 if product=-1

    for i in range(K):
        p, q, k = SET[i][0], SET[i][1], SET[i][2]
        aux = K + i
        key1 = (p, q)
        key2 = (p, k)
        key3 = (q, k)
        # for the 3 virtual spin interaction
        key4 = (p, K+i)
        key5 = (q, K+i)
        key6 = (k, K+i)
        key7 = p
        key8 = q
        key9 = k
        key10 = K+i
        sign = +1 if b[i] == 0 else -1
        J_1, J_2, h_1, h_2 = -1, -2, sign*1, sign*2
        if not key1 in pairs:
            pairs[key1] = J_1
        else:
            pairs[key1] += J_1
        if not key2 in pairs:
            pairs[key2] = J_1
        else:
            pairs[key2] += J_1
        if not key3 in pairs:
            pairs[key3] = J_1
        else:
            pairs[key3] += J_1
        if not key4 in pairs:
            pairs[key4] = J_2
        else:
            pairs[key4] += J_2
        if not key5 in pairs:
            pairs[key5] = J_2
        else:
            pairs[key5] += J_2   
        if not key6 in pairs:
            pairs[key6] = J_2
        else:
            pairs[key6] += J_2
        if not key7 in h:
            h[key7] = h_1
        else:
            h[key7] += h_1
        if not key8 in h:
            h[key8] = h_1
        else:
            h[key8] += h_1
        if not key9 in h:
            h[key9] = h_1
        else:
            h[key9] += h_1
        if not key10 in h:
            h[key10] = h_2
        else:
            h[key10] = h_2

    return pairs, h, SET, b, s_planted  # return planted solution for verification







def write_couplings(pairs, output_path):
    with open(output_path, "w") as f:
        for (p, q), Jpq in sorted(pairs.items()):
            f.write(f"{p} {q} {Jpq}\n")
    print(f"Written {len(pairs)} couplings to '{output_path}'")

def write_h(h, output_path):
    with open(output_path, "w") as f:
        for i, h_i in sorted(h.items()):
            f.write(f"{i} {h_i}\n")
    print(f"Written {len(h)} field to '{output_path}'")

def write_SET(SET, output_path):
    with open(output_path, "w") as f:
        for (i, j, k) in SET:
            f.write(f"{i} {j} {k}\n")
    print(f"Written {len(SET)} SETS to '{output_path}'")



def write_b(b, output_path):
    with open(output_path, "w") as f:
        for i in sorted(b):
            f.write(f"{i}\n")
    print(f"Written {len(b)} B to '{output_path}'")


def solve_XORSAT_gaussian(SET, b, K):
    """
    Solve the XORSAT system over GF(2) using Gaussian elimination.
    
    System: for each clause i,  x_p XOR x_q XOR x_k = b_i  (mod 2)
    where x_i = (1 - s_i) / 2  (so s=+1 → x=0, s=-1 → x=1)
    
    Returns:
        s_solution : [K] tensor of ±1 spins if a solution exists
        None       : if the system is inconsistent
    """
    import numpy as np
    
    n = K  # number of physical spin variables
    m = K  # number of equations (one per clause)
    
    # Build augmented matrix [A | b] over GF(2)
    Aug = np.zeros((m, n + 1), dtype=np.int32)
    for i in range(m):
        p, q, k = SET[i][0], SET[i][1], SET[i][2]
        Aug[i, p] = 1
        Aug[i, q] = 1
        Aug[i, k] = 1
        Aug[i, n] = int(b[i].item())  # rhs
    
    
    # Gaussian elimination over GF(2)
    pivot_cols = []
    row = 0
    for col in range(n):
        # Find pivot in this column at or below current row
        pivot = None
        for r in range(row, m):
            if Aug[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue  # free variable
        
        # Swap rows
        Aug[[row, pivot]] = Aug[[pivot, row]]
        pivot_cols.append(col)
        
        # Eliminate all other rows
        for r in range(m):
            if r != row and Aug[r, col] == 1:
                Aug[r] = (Aug[r] + Aug[row]) % 2
        row += 1
    

    
    # Check consistency: any row [0...0 | 1] is a contradiction
    for r in range(m):
        if np.all(Aug[r, :n] == 0) and Aug[r, n] == 1:
            print(f"\n❌ INCONSISTENT at row {r} — no solution exists!")
            print("   This means the random b values have no satisfying assignment.")
            return None
    
    # Back-substitute: free variables set to 0
    x = np.zeros(n, dtype=np.int32)
    for i, col in enumerate(pivot_cols):
        x[col] = Aug[i, n]
        # Subtract contributions of free variables (all set to 0, so nothing to do)
    
    # Convert x ∈ {0,1} to s ∈ {+1,-1}
    s = 1 - 2 * x
    s_tensor = torch.tensor(s, dtype=torch.float)
    
    # Verify solution
    print(f"\nSolution x: {x.tolist()}")
    print(f"Solution s: {s_tensor.tolist()}")
    
    all_ok = True
    for i in range(K):
        p, q, k = SET[i][0], SET[i][1], SET[i][2]
        product = int(s_tensor[p] * s_tensor[q] * s_tensor[k])
        expected = 1 if b[i] == 0 else -1
        if product != expected:
            all_ok = False
            print(f"  ✗ Clause {i}: s{p}*s{q}*s{k}={product} (expected {expected})")
    
    if all_ok:
        print("✓ All clauses satisfied!")
    
    return s_tensor if all_ok else None
















def generate_XORSAT1(K, rng, lam=5.0):

    SET = generate_sets(K)

    # planted solution
    x_planted = torch.randint(0, 2, (K,), generator=rng).float()
    s_planted = 1 - 2*x_planted  # convert to ±1

    b = torch.zeros(K)
    for i in range(K):
        p,q,k = SET[i]
        b[i] = (x_planted[p] + x_planted[q] + x_planted[k]) % 2

    # variables: x (K) + y (K)
    N = 2*K

    Q = torch.zeros(N, N)  # QUBO matrix

    for i in range(K):
        p,q,k = SET[i]
        y = K + i

        # build (xp + xq + xk - 2y - b)^2
        vars_ = [p,q,k,y]
        coeffs = [1,1,1,-2]

        for a in range(4):
            for b_ in range(4):
                Q[vars_[a], vars_[b_]] += lam * coeffs[a]*coeffs[b_]

        # linear shift from -b
        for a in range(4):
            Q[vars_[a], vars_[a]] += lam * (-2*b[i]*coeffs[a])

        # constant ignored

    # ── Convert QUBO → Ising ─────────────────────────────
    J = torch.zeros(N,N)
    h = torch.zeros(N)

    for i in range(N):
        for j in range(N):
            if i != j:
                J[i,j] += Q[i,j] / 4

    for i in range(N):
        h[i] += torch.sum(Q[i]) / 2

    return J, h, SET, b, s_planted


















# ── Run everything ────────────────────────────────────────────────────────────
K=99
rng = torch.Generator().manual_seed(42)
pairs, h_dict, SET, b, s_planted = generate_XORSAT(K, rng)
#J, h, SET, b, s_planted = generate_XORSAT1(K, rng)
#torch.tensor(J)
#torch.tensor(h)

# Build J and h tensors
K = len(b)
N = 2 * K

J = torch.zeros(N, N)
for (a, bb), val in pairs.items():
    J[a, bb] += val
    J[bb, a] += val
h = torch.zeros(N)
for a, val in h_dict.items():
    h[a] = val


print("=== Gaussian Elimination Solver ===")
s_gauss = solve_XORSAT_gaussian(SET, b, K)
if s_gauss==None:
    print("no solution")
if s_gauss is not None:
    # Build full config with optimal auxiliary spins
    s_full = torch.zeros(N)
    s_full[:K] = s_gauss
    for i in range(K):
        p, q, k = SET[i][0], SET[i][1], SET[i][2]
        aux = K + i
        # Try both sa values, pick best
        best_E, best_sa = float('inf'), 1.0
        for sa in [1.0, -1.0]:
            s_full[aux] = sa
            E = compute_one_energy(s_full, J, h).item()
            if E < best_E:
                best_E = E
                best_sa = sa
        s_full[aux] = best_sa
    
    E_gauss = compute_one_energy(s_full, J, h)
    print(f"\nGaussian solution energy:  {E_gauss.item():.4f}")
    print(f"Per clause:                {E_gauss.item()/K:.4f}  (expected -4.0)")
    print(f"Per spin (÷N):             {E_gauss.item()/N:.4f}  (expected -2.0)")
    
    # Compare with planted
    s_planted_full = torch.zeros(N)
    s_planted_full[:K] = s_planted
    for i in range(K):
        p, q, k = SET[i][0], SET[i][1], SET[i][2]
        aux = K + i
        best_E, best_sa = float('inf'), 1.0
        for sa in [1.0, -1.0]:
            s_planted_full[aux] = sa
            E = compute_one_energy(s_planted_full, J, h).item()
            if E < best_E:
                best_E = E
                best_sa = sa
        s_planted_full[aux] = best_sa
    
    E_planted = compute_one_energy(s_planted_full, J, h)
    print(f"\nPlanted solution energy:   {E_planted.item():.4f}")
    print(f"Match: {torch.allclose(s_gauss, s_planted) or abs(E_gauss - E_planted) < 1e-3}")

def gadget_energy(s1, s2, s3, sa, h1, h2, J1, J2):
    return (h1*(s1+s2+s3) + h2*sa 
            + J1*(s1*s2 + s2*s3 + s3*s1) 
            + J2*sa*(s1+s2+s3))



def find_gadget_params(target_configs):
    """Find (h1,h2,J1,J2) such that target_configs all have energy -4 
    and all others have energy > -4."""
    spin_values = [1, -1]
    all_3spins = list(product(spin_values, repeat=3))
    
    # Try all integer combinations in [-3,3] (paper says integer-valued)
    for h1 in range(-3,4):
        for h2 in range(-3,4):
            for J1 in range(-3,4):
                for J2 in range(-3,4):
                    valid = True
                    for spins in all_3spins:
                        s1,s2,s3 = spins
                        # minimize over sa
                        E = min(gadget_energy(s1,s2,s3,sa,h1,h2,J1,J2) 
                                for sa in [1,-1])
                        if spins in target_configs:
                            if E != -4:
                                valid = False; break
                        else:
                            if E <= -4:
                                valid = False; break
                    if valid:
                        print(f"h1={h1}, h2={h2}, J1={J1}, J2={J2}")
'''
# b=1 target: product = -1
# b=0: product = +1
target_b0 = [(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)]

# b=1: product = -1
target_b1 = [(1,1,-1), (1,-1,1), (-1,1,1), (-1,-1,-1)]
find_gadget_params(target_b1)



def find_all_valid_params():
    """Find ALL (J1,J2,h1,h2) where ground states are exactly the 4 configs 
    with product=-1, each at energy -4, and all others strictly > -4."""
    from itertools import product as iproduct
    
    gs_b1   = [(1,1,-1),(1,-1,1),(-1,1,1),(-1,-1,-1)]  # product=-1
    non_gs  = [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]   # product=+1

    valid = []
    for J1 in range(-4,5):
        for J2 in range(-4,5):
            for h1 in range(-4,5):
                for h2 in range(-4,5):
                    J_g = torch.zeros(4,4)
                    h_g = torch.zeros(4)
                    J_g[0,1]=J_g[1,0]=J_g[0,2]=J_g[2,0]=J_g[1,2]=J_g[2,1]=J1
                    J_g[0,3]=J_g[3,0]=J_g[1,3]=J_g[3,1]=J_g[2,3]=J_g[3,2]=J2
                    h_g[0]=h_g[1]=h_g[2]=h1; h_g[3]=h2
                    
                    # All gs configs must achieve exactly -4 (for optimal sa)
                    ok = True
                    for cfg in gs_b1:
                        best = min(
                            compute_energy(
                                torch.tensor(cfg+(sa,),dtype=torch.float).unsqueeze(0),
                                J_g, h_g).item()
                            for sa in [1,-1])
                        if abs(best + 4) > 1e-3:
                            ok = False; break
                    if not ok: continue
                    
                    # All non-gs configs must be strictly > -4
                    for cfg in non_gs:
                        best = min(
                            compute_energy(
                                torch.tensor(cfg+(sa,),dtype=torch.float).unsqueeze(0),
                                J_g, h_g).item()
                            for sa in [1,-1])
                        if best <= -4 + 1e-3:
                            ok = False; break
                    if ok:
                        valid.append((J1,J2,h1,h2))
                        print(f"J1={J1:+d}, J2={J2:+d}, h1={h1:+d}, h2={h2:+d}")
    return valid

print("=== Valid b=1 stored params ===")
valid_b1 = find_all_valid_params()

# Also confirm b=0
print("\n=== Confirming b=0: J1=+1,J2=+2,h1=-1,h2=-2 ===")
gs_b0 = [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]
J_g = torch.zeros(4,4); h_g = torch.zeros(4)
J_g[0,1]=J_g[1,0]=J_g[0,2]=J_g[2,0]=J_g[1,2]=J_g[2,1]=1
J_g[0,3]=J_g[3,0]=J_g[1,3]=J_g[3,1]=J_g[2,3]=J_g[3,2]=2
h_g[0]=h_g[1]=h_g[2]=-1; h_g[3]=-2
for cfg in gs_b0:
    for sa in [1,-1]:
        s = torch.tensor(cfg+(sa,),dtype=torch.float).unsqueeze(0)
        E = compute_energy(s, J_g, h_g).item()
        print(f"  {cfg+(sa,)}: E={E:.2f}")
'''
def verify_generator(L, dim, rng):
    K = L**dim - L**dim % 3
    N = 2 * K
    pairs, h_dict, SET, b,s_planted = generate_XORSAT(L, dim, rng)
    
    # Build J and h tensors
    J = torch.zeros(N, N)
    for (a, bb), val in pairs.items():
        J[a, bb] += val
        J[bb, a] += val
    h = torch.zeros(N)
    for a, val in h_dict.items():
        h[a] = val

    print(f"K={K}, N={N}")
    print(f"J range: [{J.min().item():.1f}, {J.max().item():.1f}]  (paper: [-3,3])")
    print(f"h range: [{h.min().item():.1f}, {h.max().item():.1f}]  (paper: [-3,3])")
    print(f"J symmetric: {torch.allclose(J, J.T)}")

    # ── Check each clause individually ──────────────────────────────────
    print("\n--- Per-clause gadget energy check ---")
    all_ok = True
    for i in range(K):
        p, q, k = SET[i][0], SET[i][1], SET[i][2]
        aux = K + i

        # Extract the 4x4 sub-matrix for this gadget
        idx = [p, q, k, aux]
        J_g = J[torch.tensor(idx)][:, torch.tensor(idx)]
        h_g = h[torch.tensor(idx)]

        # Brute-force minimum over 2^4 configs
        best_E = float('inf')
        best_cfg = None
        for bits in range(16):
            s = torch.tensor([(1 if (bits>>j)&1 else -1) for j in range(4)], dtype=torch.float)
            E = -0.5*(s @ J_g @ s) - (h_g @ s)
            if E < best_E:
                best_E = E
                best_cfg = s.tolist()

        expected = -4.0
        status = "✓" if abs(best_E - expected) < 1e-3 else "✗"
        if status == "✗":
            all_ok = False
            print(f"  Clause {i}: b={int(b[i].item())} spins=({p},{q},{k}) "
                  f"min_E={best_E:.2f} (expected {expected}) cfg={best_cfg}")
            print(f"    J_g=\n{J_g}")
            print(f"    h_g={h_g.tolist()}")

    if all_ok:
        print("  All clauses have correct minimum energy -4 ✓")
    
    # ── Check global minimum by brute force (only feasible for small N) ──
    if N <= 20:
        print(f"\n--- Brute force global minimum (N={N}) ---")
        best_E = float('inf')
        best_cfg = None
        for bits in range(2**N):
            s = torch.tensor([(1 if (bits>>j)&1 else -1) for j in range(N)], 
                           dtype=torch.float).unsqueeze(0)
            E = compute_one_real_energy(s[0], J, h).item()
            if E < best_E:
                best_E = E
                best_cfg = s.squeeze().tolist()

        print(f"  Global min energy:          {best_E:.4f}")
        print(f"  Global min / K (per clause):{best_E/K:.4f}  (expected -4.0)")
        print(f"  Global min / N (per spin):  {best_E/N:.4f}  (expected -2.0)")
        print(f"  Best config: {best_cfg}")

        # Verify planted solution
        print(f"\n--- Planted solution check ---")
        # Build planted config: physical spins from b, auxiliary optimized
        s_plant = torch.zeros(N)
        # ... we don't know the planted physical spins directly without generate_sets
        # but we can verify each clause's contribution
        clause_energies = []
        for i in range(K):
            p, q, k = SET[i][0], SET[i][1], SET[i][2]
            aux = K + i
            idx = torch.tensor([p, q, k, aux])
            J_g = J[idx][:, idx]
            h_g = h[idx]
            s_g = torch.tensor([best_cfg[p], best_cfg[q], best_cfg[k], best_cfg[aux]])
            E_clause = -0.5*(s_g @ J_g @ s_g) - (h_g @ s_g)
            clause_energies.append(E_clause.item())
        
        print(f"  Per-clause energies: {[f'{e:.1f}' for e in clause_energies]}")
        print(f"  Sum of clause energies: {sum(clause_energies):.4f}")
        print(f"  Are all clauses at -4? {all(abs(e+4)<1e-3 for e in clause_energies)}")

#verify_generator(L=3, dim=2, rng=torch.Generator())







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3 XORSAT interaction embeded in a easing model with auxiliary spins"
    )
    parser.add_argument('--K', type=int, default=K, help='Lattice size')

    parser.add_argument("--seed",    type=int,   default=345692,
                        help="Random seed")
    parser.add_argument("--output",  type=str,
                        default=None,
                        help="Output filename (default: auto-generated)")
    parser.add_argument("--output_path",  type=str,
                        default=None,
                        help="Output filename (default: auto-generated)")
    parser.add_argument("--output_path_path",  type=str,
                        default=None,
                        help="Output filename (default: auto-generated)")
    parser.add_argument("--output_path_path_path",  type=str,
                        default=None,
                        help="Output filename (default: auto-generated)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.output is None:
        args.output = f"couplings_XORSAT_K{args.K}_seed{args.seed}.txt"
    if args.output_path is None:
        args.output_path = f"field_XORSAT_K{args.K}_seed{args.seed}.txt"
    if args.output_path_path is None:
        args.output_path_path = f"SET_XORSAT_K{args.K}_seed{args.seed}.txt"
    if args.output_path_path_path is None:
        args.output_path_path_path = f"b_XORSAT_K{args.K}_seed{args.seed}.txt"

    #pairs,h,SET,b, s_planted= generate_XORSAT(args.L,args.dim, rng)

    write_couplings(pairs, args.output)
    write_h(h_dict, args.output_path)
    write_SET(SET, args.output_path_path)
    write_b(b, args.output_path_path_path)


if __name__ == "__main__":
    #parse all the possible arguments
    parser = argparse.ArgumentParser(description='Sequential Tempering')
    #general parsers
    parser.add_argument('--pop_size', type=int, default=100000, help='Population size')
    parser.add_argument('--K', type=int, default=7, help='Lattice size')
    parser.add_argument('--seed', type=int, default= 345692, help='Random seed')
    parser.add_argument('--Tstart', type=float, default=1.92, help='Starting temperature')
    parser.add_argument('--Tend', type=float, default=0.1, help='Ending temperature')
    parser.add_argument('--Cv_factor', type=float, default=1.618, help='Cv_factor')
    parser.add_argument('--MLMCsteps', type=int, default=5, help='Number of Machine-Learning assisted steps')
    parser.add_argument('--MCsteps', type=int, default=15, help='Number of Monte Carlo steps for each MLMC step')
    parser.add_argument('--num_temps', type=int, default=30, help='Number of annealing temperatures')
    parser.add_argument('--schedule', type=str, default="Cv_beta", help='Scheduling of temperatures')
    #specific parsers for training
    parser.add_argument('--num_epochs_start', type=int, default=40, help='Number of epochs for the first training')
    parser.add_argument('--num_epochs_retrain', type=int, default=1, help='Number of epochs for each temperature retraining')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    
    
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
    N = K*2
    seed = args.seed
    Tstart = args.Tstart
    # .cuda() can be added to the following line
    Tstart = float(Tstart)
    Tend = float(Tend)

    MLMCsteps = args.MLMCsteps
    MCsteps = args.MCsteps
    num_epochs_start = args.num_epochs_start
    num_epochs_retrain = args.num_epochs_retrain
    batch_size = args.batch_size
    observ, elapsed_time = greedy(K, torch.tensor(J),torch.tensor(h), pop_size,N,  Observables)
    minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
    real_minimum = torch.tensor(observ.get_observable_history("min_real_energy")).min()
    print( f"minimum {minimum:.5f}")
    print( f"physical minimum {real_minimum:.5f}")



