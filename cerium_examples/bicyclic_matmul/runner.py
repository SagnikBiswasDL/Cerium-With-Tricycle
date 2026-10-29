# SPDX-License-Identifier: Apache-2.0

"""Pack A and B bicyclically, encrypt them, and verify the decrypted product."""

import numpy as np
from examples.bicyclic_matmul.encoding import (BATCH, M, N, P, decode, encode, output_name,
                                               simulate)
from examples.common import SLOTS, run_cli, run_program


def make_case(verbose=True):
    """Return (raw_inputs, expected, output_scale) for run_program.

    With BICYCLIC_BATCH > 1, expected maps each output name to its values.
    """
    scale = 1 << 56
    raw_inputs, expected = {}, {}
    for i in range(BATCH):
        rng = np.random.default_rng(7 + i)
        A = rng.integers(0, 5, size=(M, N)).astype(float)
        B = rng.integers(0, 5, size=(N, P)).astype(float)
        a, b = encode(A, SLOTS), encode(B, SLOTS)

        # The NumPy model of the encrypted program must already give A @ B.
        expected[output_name(i)] = simulate(a, b)
        if not np.array_equal(decode(expected[output_name(i)], M, P), A @ B):
            raise AssertionError("bicyclic packing model does not reproduce A @ B")
        if verbose and i == 0 and M * P <= 64:
            print(f"A =\n{A}\nB =\n{B}\nA @ B =\n{A @ B}")
        raw_inputs[f"a{i or ''}"] = (a, scale)
        raw_inputs[f"b{i or ''}"] = (b, scale)

    return raw_inputs, expected if BATCH > 1 else expected["z"], scale * scale


def main(gpus):
    raw_inputs, expected, output_scale = make_case()
    if isinstance(expected, dict):
        expected = expected["z"]
    run_program("bicyclic_matmul", gpus, raw_inputs, expected, output_scale)


if __name__ == "__main__":
    run_cli(main, __doc__)
