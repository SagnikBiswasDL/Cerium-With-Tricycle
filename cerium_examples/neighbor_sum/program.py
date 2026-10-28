# SPDX-License-Identifier: Apache-2.0

"""Add each slot to its right-hand neighbour: z = x + (x << 1)."""

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, Output
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


def build_program(gpus):
    program = CeriumProgram("neighbor_sum", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            # Same input scale as the rotation example.
            x = CiphertextInput("x", scale=84, level=LEVEL)
            # Rotation needs a Galois (rotation) key; the compiler records it in
            # program_inputs and the runner generates it.
            z = x + (x << 1)
            print(f"neighbor_sum z: scale={z.scale()} bits, level={z.level()}", flush=True)
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("neighbor_sum", build_program, __doc__)
