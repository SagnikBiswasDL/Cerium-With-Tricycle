"""Latency breakdown for one compiled Cerium example (1 GPU).

Mirrors examples/common.py:run_program step by step, timing each phase with
time.perf_counter, and still checks the decrypted output. Run from
Cerium/examples with PYTHONPATH set (see run_pod.sh / env.sh):

    python bench_matmul.py bicyclic_matmul --calls 50

CERIUM_RUNTIME_NUM_ITERS=K makes each program.run() launch the compiled CUDA
graph K times back to back; GPU time per multiplication is then
run_wall / K. Prints one JSON line prefixed with "RESULT ".
"""

import argparse
import importlib
import json
import os
import random
import statistics
import time

import numpy as np

import cerium.runtime as runtime
from examples.common import SLOTS, _secret_key, main_name, output_root
from examples.primes import Primes


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("example")
    parser.add_argument("--calls", type=int, default=50, help="timed program.run() calls")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    gpus = args.gpus
    name = main_name(gpus)
    compiled = output_root(args.example) / "compiled"
    runner = importlib.import_module(f"examples.{args.example}.runner")
    raw_inputs, expected, output_scale = runner.make_case(verbose=False)
    ms = {}

    def setup_context():
        rng = random.Random(10)
        seed = np.random.default_rng(10).integers(0, 2**32, size=8, dtype=np.uint64).tolist()
        context = runtime.Context(SLOTS, Primes)
        encryptor_context = runtime.CKKSEncryptorContext(
            context, _secret_key(rng, SLOTS), _secret_key(rng, 32), seed)
        return context, encryptor_context, runtime.CKKSEncryptor(encryptor_context, seed)

    (context, encryptor_context, encryptor), ms["context_and_secret_key"] = timed(setup_context)
    inputs = runtime.RawInputsWrapper(raw_inputs)
    generator = runtime.IOGenerator(context)
    function_dir = compiled / name
    _, ms["eval_keys"] = timed(lambda: generator.generate_and_serialize_evalkeys(
        str(function_dir / "evalkeys"), str(function_dir / "program_inputs"), encryptor_context))
    _, ms["plaintexts"] = timed(lambda: generator.generate_and_serialize_plaintexts(
        str(function_dir / "plaintexts"), str(function_dir / "program_inputs"), inputs))

    program = runtime.FusedKernelProgram(context, str(compiled), name)
    functions, ms["load_program_and_build_cuda_graph"] = timed(
        lambda: program.make_program(gpus, {name: {"use_cudagraph": "True"}}, inputs))
    encrypted, ms["encrypt"] = timed(lambda: generator.generate_ciphertext_inputs(
        str(function_dir / "program_inputs"), inputs, encryptor_context))
    _, ms["copy_inputs_to_gpu"] = timed(lambda: functions[name].copy_ciphertext_inputs_to_gpu(
        str(function_dir / "inputs"), gpus, encrypted))

    _, ms["first_run"] = timed(program.run)
    runs = [timed(program.run)[1] for _ in range(args.calls)]

    if not isinstance(expected, dict):
        expected = {"z": expected}
    outputs, ms["fetch_output"] = timed(functions[name].get_program_outputs)
    actual, ms["decrypt"] = timed(lambda: {
        key: np.asarray(encryptor.decrypt_and_decode(outputs[key], output_scale)) for key in expected})
    error = max(float(np.max(np.abs(actual[k] - np.asarray(expected[k])))) for k in expected)
    passed = all(np.allclose(actual[k], expected[k], rtol=1e-5, atol=1e-4) for k in expected)

    iters = int(os.environ.get("CERIUM_RUNTIME_NUM_ITERS", "1"))
    result = {
        "example": args.example,
        "shape": getattr(runner, "SHAPE", None) or (
            os.environ.get("BICYCLIC_SHAPE") if args.example == "bicyclic_matmul"
            else os.environ.get("DIAGONAL_SIZE", "16")),
        "schedule": os.environ.get("BICYCLIC_SCHEDULE", "linear") if args.example == "bicyclic_matmul" else "-",
        "batch": len(expected),
        "graph_launches_per_run": iters,
        "run_calls": args.calls,
        "run_median_ms": statistics.median(runs),
        "run_min_ms": min(runs),
        "per_launch_ms": statistics.median(runs) / iters,
        "max_abs_error": error,
        "passed": passed,
        **{f"{k}_ms": v for k, v in ms.items()},
    }
    print("RESULT " + json.dumps(result), flush=True)
    if not passed:
        raise SystemExit(f"{args.example} output differs from its expected result")


if __name__ == "__main__":
    main()
