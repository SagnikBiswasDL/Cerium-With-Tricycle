# SPDX-License-Identifier: Apache-2.0

"""Encrypt the Tricycle layer products' inputs, run them, and check every output.

Each output is compared slot for slot with the NumPy execution of the same
program, which is itself checked against the dense block matmul first.
Ciphertexts are encrypted at 2^84 (``CIPHER_BITS``); every plaintext is encoded at the RNS prime
its product's rescale removes, so the runtime output scales are exact
(``tricycle.kernels.track_scales``).
"""

import random
import time

import numpy as np
from examples.common import SLOTS, _secret_key, main_name, output_root, run_cli
from examples.primes import Primes
from examples.tricycle_matmul.program import CIPHER_BITS, EXAMPLE, input_level, selected_matmuls
from tricycle.kernels import simulate, track_scales

INPUT_SCALE = 1 << CIPHER_BITS


def make_case(seed=11):
    """Return (raw_inputs, expected slot vectors, output scales), keyed by DSL name."""
    rng = np.random.default_rng(seed)
    raw_inputs, expected, output_scales = {}, {}, {}
    for matmul in selected_matmuls():
        left, right = matmul.random_blocks(rng)
        decoded, slots, _ = simulate(matmul, left, right, SLOTS)
        for name, reference in matmul.reference(left, right).items():
            if not np.allclose(decoded[name], reference, rtol=1e-9, atol=1e-9):
                raise AssertionError(f"{matmul.name}: slot program does not reproduce {name}")
        outputs, plaintext_scales = track_scales(matmul, Primes, INPUT_SCALE,
                                                 input_level(matmul.name), SLOTS)
        ciphertexts = set(matmul.cipher_inputs())
        for name, vector in matmul.inputs(left, right, SLOTS).items():
            scale = INPUT_SCALE if name in ciphertexts else plaintext_scales[name][0]
            raw_inputs[name] = (vector, float(scale))
        for name, tracked in outputs.items():
            expected[name] = np.asarray(slots[name])
            output_scales[name] = float(tracked.scale)
        print(f"{matmul.name}: {len(outputs)} output block(s), "
              f"{len(plaintext_scales)} plaintexts, output level {tracked.level}")
    return raw_inputs, expected, output_scales


def main(gpus, atol=1e-3):
    import cerium.runtime as runtime

    raw_inputs, expected, output_scales = make_case()
    name = main_name(gpus)
    compiled = output_root(EXAMPLE) / "compiled"
    if not (compiled / name / "program_inputs").is_file():
        raise FileNotFoundError(f"Compile {EXAMPLE} for {gpus} GPU(s) first: {compiled / name}")

    # Same setup as examples.common.run_program, which only reads output z.
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
    print(f"program.run(): {(time.perf_counter() - start) * 1e3:.1f} ms")

    outputs = functions[name].get_program_outputs()
    failures = []
    for out_name, want in expected.items():
        got = np.real(np.asarray(encryptor.decrypt_and_decode(outputs[out_name],
                                                              output_scales[out_name])))
        error = float(np.max(np.abs(got - want)))
        print(f"{out_name}: max abs error {error:.3e} (largest slot {np.max(np.abs(want)):.3f})")
        if not np.allclose(got, want, rtol=1e-4, atol=atol):
            failures.append(out_name)
    if failures:
        raise AssertionError(f"outputs differ from the NumPy program: {failures}")
    print(f"all {len(expected)} outputs match")


if __name__ == "__main__":
    run_cli(main, __doc__)
