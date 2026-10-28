# SPDX-License-Identifier: Apache-2.0

"""Plaintext matrix times encrypted vector with diagonal packing: y = W @ x."""

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, Output, PeriodicPlaintextInput
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name

# Must divide the slot count so that x repeats exactly across all slots.
SIZE = 16


def build_program(gpus):
    program = CeriumProgram("diagonal_matvec", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            # x holds the length-SIZE vector repeated across all slots. Rotate
            # at 84 bits, as in the rotation example, to keep key-switching noise small.
            x = CiphertextInput("x", scale=84, level=LEVEL)
            # diag_k[i] = W[i, (i + k) % SIZE], repeated with period SIZE.
            z = None
            for k in range(SIZE):
                diagonal = PeriodicPlaintextInput(f"diag_{k}", scale=28, level=LEVEL, period=SIZE)
                term = (x << k if k else x) * diagonal
                z = term if z is None else z + term
            print(f"diagonal_matvec {SIZE}x{SIZE}: scale={z.scale()} bits, level={z.level()}", flush=True)
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("diagonal_matvec", build_program, __doc__)
