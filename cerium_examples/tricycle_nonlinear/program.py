# SPDX-License-Identifier: Apache-2.0

"""Tricycle's nonlinear layers (``tricycle.nonlinear_kernels``) as one Cerium program.

TRICYCLE_NONLINEAR is a ``;``-separated list of kernel specs, e.g.
``gelu`` or ``gelu:layer=10,vector=8``. TRICYCLE_LIMBS is the number of
~27.6-bit primes one multiplicative level drops (2: a 2^56 scale, as in
Cerium's BERT; 3: 2^84). TRICYCLE_INPUT_LEVEL is the ciphertext input level.
"""

import argparse
import os

from cerium.dsl import CeriumFunction, CeriumProgram, Output
from examples.common import SLOTS, SUPPORTED_GPUS, main_name, output_root
from examples.primes import Primes
from tricycle.cerium_backend import CeriumBackend
from tricycle.nonlinear_kernels import make_kernel
from tricycle.poly import Evaluator

EXAMPLE = "tricycle_nonlinear"


def settings():
    limbs = int(os.environ.get("TRICYCLE_LIMBS", "2"))
    level = int(os.environ.get("TRICYCLE_INPUT_LEVEL", "21"))
    specs = [s for s in os.environ.get("TRICYCLE_NONLINEAR", "gelu").split(";") if s.strip()]
    return limbs, level, [make_kernel(s.strip()) for s in specs]


def build_program(gpus):
    limbs, level, kernels = settings()
    program = CeriumProgram(EXAMPLE, rns_bit_size=28, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            for kernel in kernels:
                backend = CeriumBackend(28 * limbs, level, 28, SLOTS)
                ev = Evaluator(backend, Primes, limbs, 28 * limbs, kernel.name)
                outputs = kernel.emit(backend, ev, level, float(2 ** (28 * limbs)))
                for name, value in outputs.items():
                    Output(name, value.handle)
                sample = next(iter(outputs.values()))
                print(f"{kernel.name}: {len(outputs)} outputs, level {level} -> {sample.level}, "
                      f"{ev.counts['ct_ct']} ct-ct and {ev.counts['ct_scalar']} ct-scalar "
                      f"multiplies per vector", flush=True)
    return program


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=SUPPORTED_GPUS, default=1)
    parser.add_argument("--vregs", type=int, default=int(os.environ.get("CERIUM_VREGS", "8192")))
    parser.add_argument("--bcus", type=int, default=int(os.environ.get("CERIUM_BCUS", "128")))
    parser.add_argument("--emit-only", action="store_true",
                        help="build the DSL graph (checks scales and levels) without compiling")
    args = parser.parse_args()
    os.environ.setdefault("CERIUM_BLOCK_SIZE", "32")
    program = build_program(args.gpus)
    if not args.emit_only:
        from cerium.compiler import cerium_compile

        cerium_compile(program, args.gpus, args.vregs, args.bcus, str(output_root(EXAMPLE)))
