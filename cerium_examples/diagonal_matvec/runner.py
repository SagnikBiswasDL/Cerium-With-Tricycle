# SPDX-License-Identifier: Apache-2.0

"""Encrypt a repeated vector, encode the matrix diagonals, and verify W @ x."""

import numpy as np
from examples.common import SLOTS, run_cli, run_program
from examples.diagonal_matvec.program import SIZE


def make_case(verbose=True):
    """Return (raw_inputs, expected, output_scale) for run_program."""
    rng = np.random.default_rng(11)
    W = rng.integers(0, 5, size=(SIZE, SIZE)).astype(float)
    x = rng.integers(0, 5, size=SIZE).astype(float)
    rows = np.arange(SIZE)
    diagonals = {f"diag_{k}": (W[rows, (rows + k) % SIZE], 1 << 28) for k in range(SIZE)}
    if verbose and SIZE <= 16:
        print(f"W @ x = {W @ x}")

    ciphertext_scale = 1 << 84
    raw_inputs = {"x": (np.tile(x, SLOTS // SIZE), ciphertext_scale), **diagonals}
    return raw_inputs, np.tile(W @ x, SLOTS // SIZE), ciphertext_scale * (1 << 28)


def main(gpus):
    run_program("diagonal_matvec", gpus, *make_case())


if __name__ == "__main__":
    run_cli(main, __doc__)
