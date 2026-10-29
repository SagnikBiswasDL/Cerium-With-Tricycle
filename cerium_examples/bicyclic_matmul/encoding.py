# SPDX-License-Identifier: Apache-2.0

"""Bicyclic packing of matrices into CKKS slot vectors, plus a NumPy model of
the encrypted computation in program.py.

Entry (i, j) of an r x c matrix with gcd(r, c) == 1 is stored in every slot k
with k % r == i and k % c == j. By the Chinese remainder theorem, slots
0 .. r*c - 1 hold each entry exactly once, and the pattern repeats every r*c
slots.

C = A @ B with A (M x N), B (N x P) and M, N, P pairwise coprime: slot k of
a * b holds A[k % M, k % N] * B[k % N, k % P], and summing the shifts
0, MP, ..., (N-1)MP leaves C[k % M, k % P] in slots k < M*P. That sum reads
slots up to M*N*P - 1 without wrapping, so one product needs M*N*P <= SLOTS.

Slot batching: M*N*P is a multiple of M, N and P, so K independent products
can occupy consecutive blocks of M*N*P slots (K * M*N*P <= SLOTS) and are
summed by the same rotations. The output of product b is in slots
b*M*N*P .. b*M*N*P + M*P - 1.

Environment overrides (read by both program.py and runner.py):
  BICYCLIC_SHAPE=M,N,P       default 3,4,5
  BICYCLIC_SCHEDULE=linear   N - 1 rotations (default)
  BICYCLIC_SCHEDULE=doubling log2(N) rotations; N must be a power of two
  BICYCLIC_SCHEDULE=hoisted  the linear rotations as one HoistedRotate
  BICYCLIC_SCHEDULE=rotacc   the linear rotations as one RotateAccumulateMany
  BICYCLIC_SCHEDULE=none     ablation: a * b and relinearization only (no rotations)
  BICYCLIC_SCHEDULE=add      ablation: a + b only (no key switching); a timing floor
  BICYCLIC_BATCH=K           K independent ciphertext pairs per program (default 1)
  BICYCLIC_SLOT_BATCH=K|max  K independent products per ciphertext (default 1;
                             1 replicates the single product across all slots)
  BICYCLIC_VALUES=int|real   entries in 0..4 (default) or uniform in [-1, 1]
"""

import os
from math import gcd

import numpy as np

from examples.common import SLOTS

# C (M x P) = A (M x N) @ B (N x P). M, N, P must be pairwise coprime.
M, N, P = (int(v) for v in os.environ.get("BICYCLIC_SHAPE", "3,4,5").split(","))
SCHEDULE = os.environ.get("BICYCLIC_SCHEDULE", "linear")
BATCH = int(os.environ.get("BICYCLIC_BATCH", "1"))
BLOCK = M * N * P
_slot_batch = os.environ.get("BICYCLIC_SLOT_BATCH", "1")
SLOT_BATCH = SLOTS // BLOCK if _slot_batch == "max" else int(_slot_batch)
VALUES = os.environ.get("BICYCLIC_VALUES", "int")
assert gcd(M, N) == gcd(N, P) == gcd(M, P) == 1, "M, N, P must be pairwise coprime"
assert SCHEDULE in ("linear", "doubling", "hoisted", "rotacc", "none", "add")
# Ablation schedules do not compute A @ B; they isolate parts of the program for timing.
ABLATION = SCHEDULE in ("none", "add")
assert SCHEDULE != "doubling" or N & (N - 1) == 0, "doubling needs N to be a power of two"
assert BATCH >= 1 and SLOT_BATCH >= 1
assert SLOT_BATCH * BLOCK <= SLOTS, f"{SLOT_BATCH} x {M}*{N}*{P} exceeds {SLOTS} slots"
assert VALUES in ("int", "real")


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


def encode_blocks(matrices):
    """Pack one matrix per slot block; a single matrix is replicated across all slots."""
    if len(matrices) == 1:
        return encode(matrices[0], SLOTS)
    vector = np.zeros(SLOTS)
    for b, matrix in enumerate(matrices):
        vector[b * BLOCK:(b + 1) * BLOCK] = encode(matrix, BLOCK)
    return vector


def decode_blocks(vector):
    """The M x P result of every slot-batched product in an output vector."""
    return [decode(np.asarray(vector)[b * BLOCK:], M, P) for b in range(SLOT_BATCH)]


def random_matrices(seed):
    """(A, B) for every slot block of one ciphertext pair."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(SLOT_BATCH):
        if VALUES == "int":
            A = rng.integers(0, 5, size=(M, N)).astype(float)
            B = rng.integers(0, 5, size=(N, P)).astype(float)
        else:
            A = rng.uniform(-1, 1, size=(M, N))
            B = rng.uniform(-1, 1, size=(N, P))
        pairs.append((A, B))
    return pairs


def rotations():
    """Left rotations that sum the N partial products of each output entry.

    linear:   z = p + (p << MP) + (p << 2MP) + ... + (p << (N-1)MP)
    doubling: z = p; z = z + (z << MP); z = z + (z << 2MP); ... (log2 N steps)
    Both give the same sum of shifts 0, MP, ..., (N-1)MP. hoisted and rotacc
    use the linear shifts.
    """
    if ABLATION:
        return []
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

    Slots k + t*M*P for t = 0 .. N-1 share k % M and k % P but cover every
    k % N, so the sum is C[k % M, k % P]. Slots outside the M*P result of
    each block hold partial sums and are not part of the result.
    """
    if SCHEDULE == "add":
        return a + b
    return accumulate(a * b, lambda x, s: np.roll(x, -s))
