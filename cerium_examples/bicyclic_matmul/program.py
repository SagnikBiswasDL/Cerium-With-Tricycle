# SPDX-License-Identifier: Apache-2.0

"""Encrypted matrix product C = A @ B with bicyclic packing."""

import argparse
import os

from cerium.dsl import (CeriumFunction, CeriumProgram, CiphertextInput, HoistedRotate, Output,
                        RotateAccumulateMany)
from examples.bicyclic_matmul.encoding import (BATCH, M, N, P, SCHEDULE, accumulate, output_name,
                                               rotations)
from examples.common import LEVEL, RNS_BITS, SUPPORTED_GPUS, main_name, output_root


def sum_rotations(product):
    if SCHEDULE == "hoisted":
        total = product
        for rotated in HoistedRotate(product, rotations()):
            total = total + rotated
        return total
    if SCHEDULE == "rotacc":
        # RotateAccumulate(product, shifts) segfaults the compiler at this commit;
        # BERT's matmul uses RotateAccumulateMany, which takes one operand per shift.
        return RotateAccumulateMany([product] * N, [0] + rotations())
    return accumulate(product, lambda x, shift: x << shift)


def build_program(gpus):
    program = CeriumProgram("bicyclic_matmul", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            for i in range(BATCH):
                # a packs A (M x N) and b packs B (N x P); see encoding.py.
                a = CiphertextInput(f"a{i or ''}", scale=56, level=LEVEL)
                b = CiphertextInput(f"b{i or ''}", scale=56, level=LEVEL)
                # One ciphertext-ciphertext product forms every A[i,j] * B[j,l].
                # Rotate at the doubled 112-bit scale so key-switching noise stays small.
                product = (a * b).relinearize()
                z = sum_rotations(product)
                Output(output_name(i), z)
            print(f"bicyclic_matmul {BATCH}x {M}x{N} @ {N}x{P} ({SCHEDULE}, {len(rotations())} rotations): "
                  f"scale={z.scale()} bits, level={z.level()}", flush=True)
    return program


if __name__ == "__main__":
    # Same as examples.common.compile_cli, but the GPU register (vregs) and
    # base-conversion-unit (bcus) counts are configurable: larger shapes exceed
    # common.py's 1024 and 128.
    from cerium.compiler import cerium_compile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=SUPPORTED_GPUS, default=1)
    parser.add_argument("--vregs", type=int, default=int(os.environ.get("CERIUM_VREGS", "1024")))
    parser.add_argument("--bcus", type=int, default=int(os.environ.get("CERIUM_BCUS", "128")))
    args = parser.parse_args()
    cerium_compile(build_program(args.gpus), args.gpus, args.vregs, args.bcus, str(output_root("bicyclic_matmul")))
