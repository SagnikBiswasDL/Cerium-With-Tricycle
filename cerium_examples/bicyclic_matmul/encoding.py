# SPDX-License-Identifier: Apache-2.0

"""Bicyclic packing of matrices into CKKS slot vectors, plus a NumPy model of
the encrypted computation in program.py.

Entry (i, j) of an r x c matrix with gcd(r, c) == 1 is stored in every slot k
with k % r == i and k % c == j. By the Chinese remainder theorem, slots
0 .. r*c - 1 hold each entry exactly once, and the pattern repeats every r*c
slots.
"""

from math import gcd

import numpy as np

# C (M x P) = A (M x N) @ B (N x P). M, N, P must be pairwise coprime.
M, N, P = 3, 4, 5
assert gcd(M, N) == gcd(N, P) == gcd(M, P) == 1


def encode(matrix, length):
    rows, cols = matrix.shape
    k = np.arange(length)
    return matrix[k % rows, k % cols].astype(float)


def decode(vector, rows, cols):
    k = np.arange(rows * cols)
    matrix = np.empty((rows, cols), dtype=vector.dtype)
    matrix[k % rows, k % cols] = vector[: rows * cols]
    return matrix


def rotations():
    """Left rotations that sum the N partial products of each output entry."""
    return [t * M * P for t in range(1, N)]


def simulate(a, b):
    """Slotwise product, then add left-rotated copies (x << s == np.roll(x, -s)).

    Slot k of a * b holds A[k % M, k % N] * B[k % N, k % P]. Slots k + t*M*P
    for t = 0 .. N-1 share k % M and k % P but cover every k % N, so the sum is
    C[k % M, k % P]. The last few slots of the full vector wrap around and are
    not part of the result.
    """
    product = a * b
    total = product.copy()
    for shift in rotations():
        total += np.roll(product, -shift)
    return total
