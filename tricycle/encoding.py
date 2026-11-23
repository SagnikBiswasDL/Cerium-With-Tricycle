"""Tricyclic slot encoding.

Slot ``k`` of an encoded tensor holds ``tensor[k % d0, k % d1, k % d2]``.
This is ``encode`` / ``decode`` from ``tricyclic_encoding.cpp`` in both
Lawrence repositories. The coprime check is the one in that file: the gcd of
every dimension together is 1, which is what the C++ ``are_coprime`` computes
(it folds ``std::gcd`` from 0). Pairwise coprime dimensions are what make the
Chinese-remainder layout one-to-one; callers that need a unique slot per
entry should use those.
"""

from math import gcd

import numpy as np


def are_coprime(dims):
    g = 0
    for dim in dims:
        g = gcd(g, int(dim))
    return g == 1


def require_coprime(dims):
    dims = tuple(int(d) for d in dims)
    if len(dims) != 3:
        raise ValueError(f"tricyclic encoding expects 3 dimensions, got {dims}")
    if not are_coprime(dims):
        raise ValueError(f"dimensions {dims} are not coprime")
    if any(d <= 0 for d in dims):
        raise ValueError(f"dimensions must be positive, got {dims}")
    return dims


def as_slot_array(values):
    """Real values stay float64; complex ones (Tricycle's packed weights) stay complex."""
    values = np.asarray(values)
    if np.iscomplexobj(values):
        return values.astype(complex)
    return values.astype(float)


def encode(tensor, shape=None):
    """Flatten ``tensor`` into the tricyclic slot order. Length is ``d0*d1*d2``."""
    tensor = as_slot_array(tensor)
    dims = require_coprime(shape if shape is not None else tensor.shape)
    if tensor.shape != dims:
        raise ValueError(f"tensor shape {tensor.shape} does not match {dims}")
    length = dims[0] * dims[1] * dims[2]
    slots = np.arange(length)
    return tensor[slots % dims[0], slots % dims[1], slots % dims[2]].copy()


def decode(encoded, shape):
    """Inverse of :func:`encode`. Entries past the first ``d0*d1*d2`` slots are ignored."""
    dims = require_coprime(shape)
    encoded = np.asarray(encoded)
    length = dims[0] * dims[1] * dims[2]
    if encoded.shape[0] < length:
        raise ValueError(f"need {length} slots to decode {dims}, got {encoded.shape[0]}")
    tensor = np.empty(dims, dtype=encoded.dtype)
    slots = np.arange(length)
    tensor[slots % dims[0], slots % dims[1], slots % dims[2]] = encoded[:length]
    return tensor


def dense_matmul(left, right):
    """Batch matmul ``(B, M, K) @ (B, K, N) -> (B, M, N)``.

    A batch of 1 on either side broadcasts, which is how Tricycle multiplies the
    ``(1, T, D)`` activations by per-head ``(H, D, K)`` weights.
    """
    left = as_slot_array(left)
    right = as_slot_array(right)
    batches_agree = left.shape[0] == right.shape[0] or 1 in (left.shape[0], right.shape[0])
    if not batches_agree or left.shape[2] != right.shape[1]:
        raise ValueError(f"cannot multiply {left.shape} by {right.shape}")
    batch = max(left.shape[0], right.shape[0])
    left = np.broadcast_to(left, (batch,) + left.shape[1:])
    right = np.broadcast_to(right, (batch,) + right.shape[1:])
    return np.einsum("bmk,bkn->bmn", left, right)


def concatenated_matmul(left, right):
    """Same product, then sum across the batch axis. Result shape is ``(1, M, N)``."""
    return dense_matmul(left, right).sum(axis=0, keepdims=True)


def split_blocks(tensor, block_dims):
    """Split a tensor into blocks the way ``split_3d_tensor`` does.

    Returns an array of shape ``(n0, n1, n2, b0, b1, b2)``.
    """
    tensor = as_slot_array(tensor)
    b0, b1, b2 = (int(d) for d in block_dims)
    d0, d1, d2 = tensor.shape
    if d0 % b0 or d1 % b1 or d2 % b2:
        raise ValueError(f"shape {tensor.shape} is not divisible by block {block_dims}")
    n0, n1, n2 = d0 // b0, d1 // b1, d2 // b2
    return np.ascontiguousarray(
        tensor.reshape(n0, b0, n1, b1, n2, b2).transpose(0, 2, 4, 1, 3, 5)
    )


def flatten_block_grid(blocks):
    """``(n0, n1, n2, ...)`` grid to the ``(n0*n1, n2)`` block matrix the tests build."""
    n0, n1, n2 = blocks.shape[:3]
    grid = [[None for _ in range(n2)] for _ in range(n0 * n1)]
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                grid[i * n1 + j][k] = blocks[i, j, k]
    return grid
