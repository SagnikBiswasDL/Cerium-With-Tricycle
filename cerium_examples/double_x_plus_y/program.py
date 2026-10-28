# SPDX-License-Identifier: Apache-2.0

"""Reuse one encrypted input twice: z = x + x + y."""

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, Output
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


def build_program(gpus):
    program = CeriumProgram("double_x_plus_y", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            x = CiphertextInput("x", scale=56, level=LEVEL)
            y = CiphertextInput("y", scale=56, level=LEVEL)
            # x appears twice in the graph but is declared, encrypted, and
            # supplied once. Both additions keep scale 2^56 and level LEVEL.
            z = x + x + y
            print(f"double_x_plus_y z: scale={z.scale()} bits, level={z.level()}", flush=True)
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("double_x_plus_y", build_program, __doc__)
