# SPDX-License-Identifier: Apache-2.0

"""Encrypt the nonlinear kernels' inputs, run them, and check every output.

Each output is compared with the NumPy execution of the same program (the
CKKS error) and with the exact function (the approximation plus CKKS error),
over the slots that hold data.
"""

import os
import random
import time

import numpy as np
from examples.common import SLOTS, _secret_key, main_name, output_root, run_cli
from examples.primes import Primes
from examples.tricycle_nonlinear.program import EXAMPLE, settings
from tricycle.nonlinear_kernels import BLOCK_SLOTS, run_numpy


def make_case(seed=11):
    limbs, level, kernels = settings()
    rng = np.random.default_rng(seed)
    raw_inputs, expected, exact, output_scales = {}, {}, {}, {}
    scale = float(2 ** (28 * limbs))
    for kernel in kernels:
        inputs = kernel.sample(rng)
        outputs, ev = run_numpy(kernel, inputs, Primes, limbs, level)
        for name, vector in inputs.items():
            raw_inputs[name] = (vector, scale)
        for name, (value, s) in ev.scalars.items():
            raw_inputs[name] = (value, s)
        reference = kernel.reference(inputs)
        for name, value in outputs.items():
            expected[name] = np.asarray(value.handle)
            exact[name] = reference[name]
            output_scales[name] = value.scale
        worst = max(float(np.max(np.abs(expected[n][:BLOCK_SLOTS] - exact[n][:BLOCK_SLOTS])))
                    for n in outputs)
        print(f"{kernel.name}: NumPy program vs exact function, max abs error {worst:.3e}")
    return raw_inputs, expected, exact, output_scales


def main(gpus):
    import cerium.runtime as runtime

    raw_inputs, expected, exact, output_scales = make_case()
    name = main_name(gpus)
    compiled = output_root(EXAMPLE) / "compiled"
    if not (compiled / name / "program_inputs").is_file():
        raise FileNotFoundError(f"Compile {EXAMPLE} for {gpus} GPU(s) first: {compiled / name}")

    rng = random.Random(10)
    seed = np.random.default_rng(10).integers(0, 2**32, size=8, dtype=np.uint64).tolist()
    context = runtime.Context(SLOTS, Primes)
    encryptor_context = runtime.CKKSEncryptorContext(
        context, _secret_key(rng, SLOTS), _secret_key(rng, 32), seed
    )
    encryptor = runtime.CKKSEncryptor(encryptor_context, seed)
    inputs = runtime.RawInputsWrapper(raw_inputs)
    generator = runtime.IOGenerator(context)
    for function_dir in sorted(compiled.glob(f"*_{gpus}gpus")):
        function_inputs = function_dir / "program_inputs"
        if not function_inputs.is_file():
            continue
        generator.generate_and_serialize_evalkeys(
            str(function_dir / "evalkeys"), str(function_inputs), encryptor_context
        )
        generator.generate_and_serialize_plaintexts(
            str(function_dir / "plaintexts"), str(function_inputs), inputs
        )
    program = runtime.FusedKernelProgram(context, str(compiled), name)
    functions = program.make_program(gpus, {name: {"use_cudagraph": "True"}}, inputs)
    encrypted_inputs = generator.generate_ciphertext_inputs(
        str(compiled / name / "program_inputs"), inputs, encryptor_context
    )
    functions[name].copy_ciphertext_inputs_to_gpu(
        str(compiled / name / "inputs"), gpus, encrypted_inputs
    )
    start = time.perf_counter()
    program.run()
    print(f"program.run(): {(time.perf_counter() - start) * 1e3:.1f} ms (first, cold)")
    warmup = int(os.environ.get("TRICYCLE_WARMUP", "3"))
    reps = int(os.environ.get("TRICYCLE_REPS", "10"))
    for _ in range(warmup):
        program.run()
    runs = []
    for _ in range(reps):
        start = time.perf_counter()
        program.run()
        runs.append((time.perf_counter() - start) * 1e3)
    if runs:
        print(f"program.run() warm: median {np.median(runs):.1f} ms, "
              f"p95 {np.percentile(runs, 95):.1f} ms, min {min(runs):.1f} ms "
              f"({warmup} warmup, {reps} timed)")

    outputs = functions[name].get_program_outputs()
    worst_ckks, worst_exact = 0.0, 0.0
    for out_name, want in expected.items():
        got = np.real(np.asarray(encryptor.decrypt_and_decode(outputs[out_name],
                                                              output_scales[out_name])))
        ckks = float(np.max(np.abs(got[:BLOCK_SLOTS] - want[:BLOCK_SLOTS])))
        total = float(np.max(np.abs(got[:BLOCK_SLOTS] - exact[out_name][:BLOCK_SLOTS])))
        worst_ckks, worst_exact = max(worst_ckks, ckks), max(worst_exact, total)
        print(f"{out_name}: vs NumPy program {ckks:.3e}, vs exact function {total:.3e} "
              f"(largest |value| {np.max(np.abs(want[:BLOCK_SLOTS])):.3f})")
    print(f"worst over {len(expected)} outputs: vs NumPy program {worst_ckks:.3e}, "
          f"vs exact function {worst_exact:.3e}")


if __name__ == "__main__":
    run_cli(main, __doc__)
