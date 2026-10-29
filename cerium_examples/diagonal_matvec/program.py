# SPDX-License-Identifier: Apache-2.0

"""Plaintext matrix times encrypted vector with diagonal packing: y = W @ x."""

import argparse
import os

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, Output, PeriodicPlaintextInput
from examples.common import LEVEL, RNS_BITS, SLOTS, SUPPORTED_GPUS, main_name, output_root

# Must divide the slot count so that x repeats exactly across all slots.
SIZE = int(os.environ.get("DIAGONAL_SIZE", "16"))
assert SLOTS % SIZE == 0


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
    # Same as examples.common.compile_cli, but the GPU register (vregs) and
    # base-conversion-unit (bcus) counts are configurable: larger sizes exceed
    # common.py's 1024 and 128.
    from cerium.compiler import cerium_compile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=SUPPORTED_GPUS, default=1)
    parser.add_argument("--vregs", type=int, default=int(os.environ.get("CERIUM_VREGS", "1024")))
    parser.add_argument("--bcus", type=int, default=int(os.environ.get("CERIUM_BCUS", "128")))
    args = parser.parse_args()
    cerium_compile(build_program(args.gpus), args.gpus, args.vregs, args.bcus, str(output_root("diagonal_matvec")))
