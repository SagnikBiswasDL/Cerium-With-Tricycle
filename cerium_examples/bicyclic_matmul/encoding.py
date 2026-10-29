# SPDX-License-Identifier: Apache-2.0

"""Bicyclic packing of matrices into CKKS slot vectors, plus a NumPy model of
the encrypted computation in program.py.

Entry (i, j) of an r x c matrix with gcd(r, c) == 1 is stored in every slot k
with k % r == i and k % c == j. By the Chinese remainder theorem, slots
0 .. r*c - 1 hold each entry exactly once, and the pattern repeats every r*c
slots.

Environment overrides (read by both program.py and runner.py):
  BICYCLIC_SHAPE=M,N,P       default 3,4,5
  BICYCLIC_SCHEDULE=linear   N - 1 rotations (default)
  BICYCLIC_SCHEDULE=doubling log2(N) rotations; N must be a power of two
  BICYCLIC_SCHEDULE=hoisted  the linear rotations as one HoistedRotate
  BICYCLIC_SCHEDULE=rotacc   the linear rotations as one RotateAccumulateMany
  BICYCLIC_BATCH=K           K independent products in one program (default 1)
"""

import os
from math import gcd

import numpy as np

# C (M x P) = A (M x N) @ B (N x P). M, N, P must be pairwise coprime.
M, N, P = (int(v) for v in os.environ.get("BICYCLIC_SHAPE", "3,4,5").split(","))
SCHEDULE = os.environ.get("BICYCLIC_SCHEDULE", "linear")
BATCH = int(os.environ.get("BICYCLIC_BATCH", "1"))
assert gcd(M, N) == gcd(N, P) == gcd(M, P) == 1
assert SCHEDULE in ("linear", "doubling", "hoisted", "rotacc")
assert BATCH >= 1
assert SCHEDULE == "linear" or N & (N - 1) == 0, "doubling needs N to be a power of two"


def output_name(index):
    return f"z{index or ''}"


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
    """Left rotations that sum the N partial products of each output entry.

    linear:   z = p + (p << MP) + (p << 2MP) + ... + (p << (N-1)MP)
    doubling: z = p; z = z + (z << MP); z = z + (z << 2MP); ... (log2 N steps)
    Both give the same sum of shifts 0, MP, ..., (N-1)MP. hoisted and rotacc
    use the linear shifts.
    """
    if SCHEDULE != "doubling":
        return [t * M * P for t in range(1, N)]
    return [step * M * P for step in (1 << i for i in range(N.bit_length() - 1))]


def accumulate(product, rotate):
    """Apply the rotation schedule; rotate(x, s) is a left rotation by s."""
    if SCHEDULE != "doubling":
        total = product
        for shift in rotations():
            total = total + rotate(product, shift)
        return total
    total = product
    for shift in rotations():
        total = total + rotate(total, shift)
    return total


def simulate(a, b):
    """NumPy model of program.py (x << s == np.roll(x, -s)).

    Slot k of a * b holds A[k % M, k % N] * B[k % N, k % P]. Slots k + t*M*P
    for t = 0 .. N-1 share k % M and k % P but cover every k % N, so the sum is
    C[k % M, k % P]. The last (N-1)*M*P slots wrap around and are not part of
    the result.
    """
    return accumulate(a * b, lambda x, s: np.roll(x, -s))
