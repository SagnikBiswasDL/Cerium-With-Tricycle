# SPDX-License-Identifier: Apache-2.0

"""Encrypt a repeated vector, encode the matrix diagonals, and verify W @ x."""

import numpy as np
from examples.common import SLOTS, run_cli, run_program
from examples.diagonal_matvec.program import SIZE


def main(gpus):
    rng = np.random.default_rng(11)
    W = rng.integers(0, 5, size=(SIZE, SIZE)).astype(float)
    x = rng.integers(0, 5, size=SIZE).astype(float)
    rows = np.arange(SIZE)
    diagonals = {f"diag_{k}": (W[rows, (rows + k) % SIZE], 1 << 28) for k in range(SIZE)}
    print(f"W @ x = {W @ x}")

    ciphertext_scale = 1 << 84
    run_program("diagonal_matvec", gpus,
                {"x": (np.tile(x, SLOTS // SIZE), ciphertext_scale), **diagonals},
                np.tile(W @ x, SLOTS // SIZE), ciphertext_scale * (1 << 28))


if __name__ == "__main__":
    run_cli(main, __doc__)
