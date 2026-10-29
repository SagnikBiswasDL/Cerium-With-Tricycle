# SPDX-License-Identifier: Apache-2.0

"""Pack A and B bicyclically, encrypt them, and verify the decrypted product."""

import numpy as np
from examples.bicyclic_matmul.encoding import (ABLATION, BATCH, M, N, P, SCHEDULE, SLOT_BATCH,
                                               decode_blocks, encode_blocks, output_name,
                                               random_matrices, simulate)
from examples.common import run_cli, run_program

INPUT_SCALE = 1 << 56


def references():
    """{output name: [A_b @ B_b for each slot block]} computed in NumPy."""
    return {output_name(i): [A @ B for A, B in random_matrices(7 + i)] for i in range(BATCH)}


def make_case(verbose=True):
    """Return (raw_inputs, expected, output_scale) for run_program.

    expected holds the full slot vectors predicted by the NumPy model of the
    encrypted program; with BICYCLIC_BATCH > 1 it maps each output name to its
    vector.
    """
    raw_inputs, expected = {}, {}
    for i in range(BATCH):
        pairs = random_matrices(7 + i)
        a = encode_blocks([A for A, _ in pairs])
        b = encode_blocks([B for _, B in pairs])
        name = output_name(i)
        expected[name] = simulate(a, b)
        # The NumPy model of the encrypted program must already give A @ B.
        for got, (A, B) in zip([] if ABLATION else decode_blocks(expected[name]), pairs):
            if not np.allclose(got, A @ B, rtol=1e-12, atol=1e-12):
                raise AssertionError("bicyclic packing model does not reproduce A @ B")
        if verbose and i == 0 and M * P <= 64:
            A, B = pairs[0]
            print(f"A =\n{A}\nB =\n{B}\nA @ B =\n{A @ B}")
        raw_inputs[f"a{i or ''}"] = (a, INPUT_SCALE)
        raw_inputs[f"b{i or ''}"] = (b, INPUT_SCALE)

    if verbose and SLOT_BATCH > 1:
        print(f"{SLOT_BATCH} independent {M}x{N} @ {N}x{P} products per ciphertext")
    output_scale = INPUT_SCALE if SCHEDULE == "add" else INPUT_SCALE * INPUT_SCALE
    return raw_inputs, expected if BATCH > 1 else expected["z"], output_scale


def main(gpus):
    raw_inputs, expected, output_scale = make_case()
    if isinstance(expected, dict):
        expected = expected["z"]
    run_program("bicyclic_matmul", gpus, raw_inputs, expected, output_scale)


if __name__ == "__main__":
    run_cli(main, __doc__)
