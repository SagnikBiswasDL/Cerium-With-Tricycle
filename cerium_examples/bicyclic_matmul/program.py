# SPDX-License-Identifier: Apache-2.0

"""Encrypted matrix product C = A @ B with bicyclic packing."""

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, Output
from examples.bicyclic_matmul.encoding import M, N, P, rotations
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


def build_program(gpus):
    program = CeriumProgram("bicyclic_matmul", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            # a packs A (M x N) and b packs B (N x P); see encoding.py.
            a = CiphertextInput("a", scale=56, level=LEVEL)
            b = CiphertextInput("b", scale=56, level=LEVEL)
            # One ciphertext-ciphertext product forms every A[i,j] * B[j,l].
            # Rotate at the doubled 112-bit scale so key-switching noise stays small.
            product = (a * b).relinearize()
            z = product
            for shift in rotations():
                z = z + (product << shift)
            print(f"bicyclic_matmul {M}x{N} @ {N}x{P}: scale={z.scale()} bits, level={z.level()}", flush=True)
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("bicyclic_matmul", build_program, __doc__)
