# Instance File Format

This document describes the simple text-based format used for storing spin glass instances in this repository.

## Format Specification

An instance file is a plain text file with the following structure:

1.  The first line contains two space-separated integers:
    -   `N`: The number of spins in the system.
    -   `M`: The number of non-zero couplings (edges) in the graph.

2.  The following `M` lines each describe a coupling, with three space-separated values:
    -   `i`: The index of the first spin (1-based or 0-based, should be consistent).
    -   `j`: The index of the second spin.
    -   `J_ij`: The value of the coupling constant between spin `i` and `j`. This is a floating-point number.

## Example

Here is an example of a small instance with 4 spins and 4 couplings:

```
4 4
0 1 1.0
0 2 -1.0
1 3 -1.0
2 3 1.0
```

This represents a system with spins {0, 1, 2, 3} and the Hamiltonian:
H = - (1.0 * s_0 * s_1 - 1.0 * s_0 * s_2 - 1.0 * s_1 * s_3 + 1.0 * s_2 * s_3)
