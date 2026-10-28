# SPDX-License-Identifier: Apache-2.0

"""Weighted sum of two encrypted vectors with plaintext weights: z = x * a + y * b."""

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, PlaintextInput, Output
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


def build_program(gpus):
    program = CeriumProgram("weighted_sum", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            x = CiphertextInput("x", scale=28, level=LEVEL)
            y = CiphertextInput("y", scale=28, level=LEVEL)
            a = PlaintextInput("a", scale=28, level=LEVEL)
            b = PlaintextInput("b", scale=28, level=LEVEL)
            # Each ciphertext-plaintext product has scale 28 + 28 = 56 bits, so
            # the two products can be added without rescaling.
            z = x * a + y * b
            print(f"weighted_sum z: scale={z.scale()} bits, level={z.level()}", flush=True)
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("weighted_sum", build_program, __doc__)
