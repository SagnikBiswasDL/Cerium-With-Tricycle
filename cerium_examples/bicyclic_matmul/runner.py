# SPDX-License-Identifier: Apache-2.0

"""Pack A and B bicyclically, encrypt them, and verify the decrypted product."""

import numpy as np
from examples.bicyclic_matmul.encoding import M, N, P, decode, encode, simulate
from examples.common import SLOTS, run_cli, run_program


def main(gpus):
    rng = np.random.default_rng(7)
    A = rng.integers(0, 5, size=(M, N)).astype(float)
    B = rng.integers(0, 5, size=(N, P)).astype(float)
    a, b = encode(A, SLOTS), encode(B, SLOTS)

    # The NumPy model of the encrypted program must already give A @ B.
    expected = simulate(a, b)
    if not np.array_equal(decode(expected, M, P), A @ B):
        raise AssertionError("bicyclic packing model does not reproduce A @ B")
    print(f"A =\n{A}\nB =\n{B}\nA @ B =\n{A @ B}")

    scale = 1 << 56
    run_program("bicyclic_matmul", gpus, {"a": (a, scale), "b": (b, scale)},
                expected, scale * scale)


if __name__ == "__main__":
    run_cli(main, __doc__)
