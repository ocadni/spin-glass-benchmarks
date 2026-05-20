
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
#from greedy_XORSAT import *

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


def generate_XORSAT(K):

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

def write(h,pairs,seed,output_path):
    with open(output_path, "w") as f:
        f.write(f"# model=xorsat K={len(h)/2} seed={seed}\n")
        for i, h_i in sorted(h.items()):
            f.write(f"{i} {h_i}\n")
        for (p, q), Jpq in sorted(pairs.items()):
            f.write(f"{p} {q} {Jpq}\n")
    print(f"Written {len(h)} field to '{output_path}'")
    print(f"Written {len(pairs)} couplings to '{output_path}'")