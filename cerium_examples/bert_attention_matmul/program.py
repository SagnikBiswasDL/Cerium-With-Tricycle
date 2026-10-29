# SPDX-License-Identifier: Apache-2.0

"""BERT's encrypted attention matmul, compiled on its own.

The body is matmul_ct128x128_ct128x128_function from
benchmarks/bert_base/bert_cerium.py, with its function arguments turned into
program inputs so it can be built and timed like the other examples. It
computes two heads of P_h (128x128) @ V_h (128x64); see packing.py.
"""

import argparse
import os

from cerium.dsl import (BreakVector, BsgsMultiplyAccumulate, CeriumFunction, CeriumProgram,
                        CiphertextInput, HoistedRotate, MakeVector, Output,
                        PeriodicPlaintextInput, RotateAccumulateMany)
from examples.common import RNS_BITS, SUPPORTED_GPUS, main_name, output_root

SCALE = RNS_BITS
PROBS_LEVEL = 8
VALUES_LEVEL = 9


def build_program(gpus):
    program = CeriumProgram("bert_attention_matmul", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            ct1 = CiphertextInput("probs", 3 * SCALE, PROBS_LEVEL)
            ct2 = CiphertextInput("values", 3 * SCALE, VALUES_LEVEL)
            babySteps = [i * 256 for i in range(8)]
            giantSteps = [i * 256 * 8 for i in range(0, 16)]
            pts = [PeriodicPlaintextInput(f"attention_matmul_mask_128x128_rot_{i*128}", SCALE + SCALE // 2, ct2.level(), 128) for i in range(0, 128)]
            diag = BsgsMultiplyAccumulate(ct2, pts, babySteps, giantSteps, rescaleLevels=1)

            ct1_gs = HoistedRotate(ct1, [(8 * gs - 2048 * gs) for gs in range(16)])
            diag_bs = HoistedRotate(diag, [(256 * bs) for bs in range(8)])

            diag_bs_vec = MakeVector(diag_bs)
            diag_bs_vec = diag_bs_vec.doubleRescale()

            babysteps_accumulated = [None for _ in range(16)]

            for gs in range(0, 128 // 8, 1):
                l = MakeVector(HoistedRotate(ct1_gs[gs], [bs for bs in range(8)]))
                r = MakeVector(HoistedRotate(ct1_gs[gs], [(-128 + bs) for bs in range(8)]))
                diagRot = diag_bs_vec
                maskL = MakeVector([PeriodicPlaintextInput(f"attention_matmul_maskL_128x128_{8*gs + bs}", SCALE + SCALE // 2, l.level(), 128) for bs in range(8)])
                maskR = MakeVector([PeriodicPlaintextInput(f"attention_matmul_maskR_128x128_{8*gs + bs}", SCALE + SCALE // 2, r.level(), 128) for bs in range(8)])
                vec = l * maskL + r * maskR
                vec = vec.doubleRescale()
                vec = vec * diagRot
                vec = BreakVector(vec)
                sumbs = vec[0]
                for bs in range(1, 8):
                    sumbs += vec[bs]
                babysteps_accumulated[gs] = sumbs

            babysteps_accumulated = MakeVector(babysteps_accumulated)
            babysteps_accumulated = babysteps_accumulated.relinearize()
            babysteps_accumulated = BreakVector(babysteps_accumulated)

            sumi = RotateAccumulateMany(babysteps_accumulated, [256 * 8 * gs for gs in range(16)])

            sumi = sumi.rescale()
            print(f"bert_attention_matmul: scale={sumi.scale()} bits, level={sumi.level()}", flush=True)
            Output("z", sumi)
    return program


if __name__ == "__main__":
    from cerium.compiler import cerium_compile

    # bert_cerium.py builds this function with block size 256, 3096 vregs and 128 bcus.
    os.environ.setdefault("CERIUM_BLOCK_SIZE", "256")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=SUPPORTED_GPUS, default=1)
    parser.add_argument("--vregs", type=int, default=int(os.environ.get("CERIUM_VREGS", "3096")))
    parser.add_argument("--bcus", type=int, default=int(os.environ.get("CERIUM_BCUS", "128")))
    args = parser.parse_args()
    cerium_compile(build_program(args.gpus), args.gpus, args.vregs, args.bcus, str(output_root("bert_attention_matmul")))
