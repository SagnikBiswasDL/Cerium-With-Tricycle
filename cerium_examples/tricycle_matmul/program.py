# SPDX-License-Identifier: Apache-2.0

"""Every matrix product of one Tricycle encoder layer, as one Cerium program.

The seven products of ``evaluate_layer`` in Tricycle's ``transformer.cpp``:
QKV, the K/V reshape, Q @ K^T, softmax @ V, the attention output projection,
FF1 and FF2. Each is lowered by ``tricycle.kernels`` (the same program the
NumPy tests execute) through ``tricycle.cerium_backend``.

TRICYCLE_SHAPE selects ``mini`` (default), ``tiny`` (BERT-Tiny) or ``base``
(BERT-Base). TRICYCLE_MATMULS is a comma-separated subset of
``qkv,reshape,qk,pv,attn_out,ff1,ff2`` (default: all). TRICYCLE_FUSED picks
the plaintext products emitted with Cerium's BsgsMultiplyAccumulate.
"""

import argparse
import os

from cerium.dsl import CeriumFunction, CeriumProgram, Output
from examples.common import LEVEL, RNS_BITS, SLOTS, SUPPORTED_GPUS, main_name, output_root
from tricycle.cerium_backend import CeriumBackend
from tricycle.kernels import SHAPES, tricycle_layer_matmuls

EXAMPLE = "tricycle_matmul"
# Cerium's key-switching noise is about 2^52 (familiarization/NOTES.md): at a
# 2^56 scale every rotation costs ~0.06 of error; at 2^84 it costs ~2e-10.
CIPHER_BITS = 3 * RNS_BITS


def selected_matmuls():
    """TRICYCLE_FUSED lists plaintext products to emit as BsgsMultiplyAccumulate
    over ciphertext vectors (``all`` for every one), e.g. ``qkv,ff1,ff2``."""
    shape = SHAPES[os.environ.get("TRICYCLE_SHAPE", "mini")]
    matmuls = tricycle_layer_matmuls(shape, SLOTS)
    fused = {name.strip() for name in os.environ.get("TRICYCLE_FUSED", "").split(",")}
    for matmul in matmuls:
        matmul.fused = matmul.kind != "ct_ct" and ("all" in fused or matmul.name in fused)
    wanted = os.environ.get("TRICYCLE_MATMULS")
    if not wanted:
        return matmuls
    names = [name.strip() for name in wanted.split(",") if name.strip()]
    known = {m.name for m in matmuls}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"unknown matmuls {unknown}; choose from {sorted(known)}")
    return [m for m in matmuls if m.name in names]


def input_level(name):
    """Ciphertext input level for one product.

    TRICYCLE_LEVELS overrides it per product, e.g. ``attn_out=5,ff1=5,ff2=5``:
    the levels Cerium's own BERT-Base feeds its attention-output and
    feed-forward matmuls. Everything else starts at LEVEL.
    """
    levels = {}
    for item in os.environ.get("TRICYCLE_LEVELS", "").split(","):
        if item.strip():
            key, value = item.split("=")
            levels[key.strip()] = int(value)
    return levels.get(name, LEVEL)


def build_program(gpus):
    program = CeriumProgram(EXAMPLE, rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            for matmul in selected_matmuls():
                backend = CeriumBackend(CIPHER_BITS, input_level(matmul.name), RNS_BITS, SLOTS)
                outputs = matmul.emit(backend, SLOTS)
                for name, value in outputs.items():
                    Output(name, value)
                plan = matmul.plan
                sample = next(iter(outputs.values()))
                print(f"{matmul.name}: {matmul.description}\n"
                      f"    blocks {matmul.left_grid} x {matmul.right_grid} -> {matmul.out_grid}, "
                      f"{plan.num_baby_steps} baby x {len(plan.chunks)} giant steps, "
                      f"output scale={sample.scale()} bits, level={sample.level()}", flush=True)
    return program


if __name__ == "__main__":
    # examples.common.compile_cli with configurable GPU register (vregs) and
    # base-conversion-unit (bcus) counts, as in bicyclic_matmul. common.py's
    # 1024 vregs make the register allocator fail ("Called Evict Reg") even on
    # the mini shape.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=SUPPORTED_GPUS, default=1)
    parser.add_argument("--vregs", type=int, default=int(os.environ.get("CERIUM_VREGS", "8192")))
    parser.add_argument("--bcus", type=int, default=int(os.environ.get("CERIUM_BCUS", "256")))
    parser.add_argument("--emit-only", action="store_true",
                        help="build the DSL graph (checks scales and levels) without compiling")
    args = parser.parse_args()
    # Cerium's BERT-Base compiles its weight matmuls with block size 32
    # (benchmarks/bert_base/bert_cerium.py). The default 1024 makes the
    # plaintext multiply-accumulates ~10x slower here (README, v3_blocksize logs).
    os.environ.setdefault("CERIUM_BLOCK_SIZE", "32")
    program = build_program(args.gpus)
    if not args.emit_only:
        from cerium.compiler import cerium_compile

        cerium_compile(program, args.gpus, args.vregs, args.bcus, str(output_root(EXAMPLE)))
