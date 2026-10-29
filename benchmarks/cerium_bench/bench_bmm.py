"""Benchmark the bicyclic ciphertext x ciphertext matrix product (BMM) in Cerium.

Run from Cerium/examples with the Cerium virtualenv and PYTHONPATH (env.sh).
The matrix shape, rotation schedule and batching are read from the
BICYCLIC_* environment variables (examples/bicyclic_matmul/encoding.py).

    python bench_bmm.py compile [--vregs 8192 --bcus 256]
    python bench_bmm.py run [--warmup 10 --reps 200]

compile
    dsl_build_ms      Python DSL builds the operation graph (also counts source ops)
    cerium_compile_ms Cerium compiler: graph passes, scheduling, CUDA code generation
    nvcc_build_ms     make_example.sh: nvcc build of the generated kernels + staging
  Writes outputs/bicyclic_matmul/compile_record.json with source-level and
  compiled-level operation counts.

run (every phase timed with time.perf_counter on the host)
    keygen_secret_ms      CKKS context + secret/ephemeral keys + encryptor (CPU)
    keygen_eval_ms        relinearization/rotation key generation, serialized to disk (CPU)
    encode_plaintexts_ms  plaintext operands encoded and serialized (none for BMM)
    load_program_ms       read keys/plaintexts, copy them to the GPU, build the CUDA graph
    encrypt_ms            encode + encrypt the input vectors (CPU)
    h2d_inputs_ms         copy input ciphertexts to the GPU (blocking cudaMemcpy)
    run                   one program.run(): launches the CUDA graph
                          CERIUM_RUNTIME_NUM_ITERS times on a new stream and returns
                          after cudaDeviceSynchronize, so the GPU work is complete.
                          Measured over --reps calls after --warmup untimed calls.
    d2h_outputs_ms        copy output ciphertexts to the host (ends with cudaDeviceSynchronize)
    decrypt_ms            decrypt + decode every output (CPU)
"""

import argparse
import importlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_info import ckks_parameters, environment  # noqa: E402
from opcount import count_compiled, count_source  # noqa: E402

EXAMPLE = "bicyclic_matmul"


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1e3


def latency_stats(samples):
    a = np.asarray(samples)
    return {
        "n": len(samples),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "mean": float(a.mean()),
        "stdev": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }


def case_description():
    enc = importlib.import_module(f"examples.{EXAMPLE}.encoding")
    from examples.common import SLOTS
    useful_inputs = enc.SLOT_BATCH * enc.BLOCK
    return {
        "shape": f"{enc.M}x{enc.N} @ {enc.N}x{enc.P}",
        "M": enc.M, "N": enc.N, "P": enc.P,
        "schedule": enc.SCHEDULE,
        "ciphertext_pairs": enc.BATCH,
        "slot_batch": enc.SLOT_BATCH,
        "slot_block": enc.BLOCK,
        "values": enc.VALUES,
        "rotations_in_schedule": len(enc.rotations()),
        # Slots of each input ciphertext holding a distinct product's data
        # (M*N*P per product). A single product is replicated to fill all
        # slots, but only its first block is useful.
        "useful_slot_occupancy": useful_inputs / SLOTS,
        "output_entries_per_ciphertext": enc.SLOT_BATCH * enc.M * enc.P,
        "multiply_adds_total": enc.BATCH * enc.SLOT_BATCH * enc.M * enc.N * enc.P,
    }


def compile_case(args):
    from cerium.compiler import cerium_compile
    from examples.common import output_root
    program_module = importlib.import_module(f"examples.{EXAMPLE}.program")
    root = output_root(EXAMPLE)
    record = {"case": case_description(), "vregs": args.vregs, "bcus": args.bcus}

    with count_source() as tracker:
        program, record["dsl_build_ms"] = timed(lambda: program_module.build_program(1))
    record["source_ops"] = dict(tracker.counts)

    os.environ["CERIUM_PRINT_GRAPHS"] = "1"
    _, record["cerium_compile_ms"] = timed(
        lambda: cerium_compile(program, 1, args.vregs, args.bcus, str(root)))
    graphs = root / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    for dot in Path(".").glob("main_1gpus_cerium_graph*"):
        shutil.move(str(dot), graphs / dot.name)

    build, record["nvcc_build_ms"] = timed(lambda: subprocess.run(
        ["bash", "make_example.sh", EXAMPLE, "1"], capture_output=True, text=True))
    if build.returncode != 0:
        sys.stderr.write(build.stdout[-4000:] + build.stderr[-4000:])
        raise SystemExit("make_example.sh failed")
    record["compiled_ops"] = count_compiled(
        graphs / "main_1gpus_cerium_graph_simple.dot",
        root / "main_1gpus" / "partition_0" / "ll.functions")
    record["compiled_ops_after_horizontal_fusion"] = count_compiled(
        graphs / "main_1gpus_cerium_graph_horizontal.dot")
    return record, root


def cmd_compile(args):
    (record, root), total_ms = timed(lambda: compile_case(args))
    record["compile_total_ms"] = total_ms
    (root / "compile_record.json").write_text(json.dumps(record, indent=1))
    print("\nCOMPILE " + json.dumps(record), flush=True)


def errors(actual, enc, runner):
    """Error of the decrypted useful outputs against NumPy A @ B.

    max_abs_error    max |C_hat - C| over every useful output entry
    rel_fro_error    ||C_hat - C||_F / ||C||_F, all products stacked together
    max_abs_rel_to_max  max |C_hat - C| / max |C|
    model_max_abs_error max error over all slots against the NumPy model of the
                     encrypted program (includes non-output slots)
    """
    out = {}
    if not enc.ABLATION:
        references = runner.references()
        got, want = [], []
        for name, refs in references.items():
            got += [m.ravel() for m in enc.decode_blocks(actual[name])]
            want += [r.ravel() for r in refs]
        got, want = np.concatenate(got), np.concatenate(want)
        diff = got - want
        out.update({
            "max_abs_error": float(np.max(np.abs(diff))),
            "rel_fro_error": float(np.linalg.norm(diff) / np.linalg.norm(want)),
            "max_abs_rel_to_max": float(np.max(np.abs(diff)) / np.max(np.abs(want))),
            "useful_output_entries": int(want.size),
            "reference_max_abs": float(np.max(np.abs(want))),
        })
    return out


def cmd_run(args):
    import cerium.runtime as runtime
    from examples.common import LEVEL, SLOTS, _secret_key, main_name, output_root
    from examples.primes import Primes

    enc = importlib.import_module(f"examples.{EXAMPLE}.encoding")
    runner = importlib.import_module(f"examples.{EXAMPLE}.runner")
    name = main_name(1)
    root = output_root(EXAMPLE)
    compiled = root / "compiled"
    function_dir = compiled / name
    raw_inputs, expected, output_scale = runner.make_case(verbose=False)
    if not isinstance(expected, dict):
        expected = {"z": expected}
    ms = {}

    def setup_context():
        rng = random.Random(10)
        seed = np.random.default_rng(10).integers(0, 2**32, size=8, dtype=np.uint64).tolist()
        context = runtime.Context(SLOTS, Primes)
        encryptor_context = runtime.CKKSEncryptorContext(
            context, _secret_key(rng, SLOTS), _secret_key(rng, 32), seed)
        return context, encryptor_context, runtime.CKKSEncryptor(encryptor_context, seed)

    (context, encryptor_context, encryptor), ms["keygen_secret_ms"] = timed(setup_context)
    inputs = runtime.RawInputsWrapper(raw_inputs)
    generator = runtime.IOGenerator(context)
    _, ms["keygen_eval_ms"] = timed(lambda: generator.generate_and_serialize_evalkeys(
        str(function_dir / "evalkeys"), str(function_dir / "program_inputs"), encryptor_context))
    _, ms["encode_plaintexts_ms"] = timed(lambda: generator.generate_and_serialize_plaintexts(
        str(function_dir / "plaintexts"), str(function_dir / "program_inputs"), inputs))
    program = runtime.FusedKernelProgram(context, str(compiled), name)
    functions, ms["load_program_ms"] = timed(
        lambda: program.make_program(1, {name: {"use_cudagraph": "True"}}, inputs))
    encrypted, ms["encrypt_ms"] = timed(lambda: generator.generate_ciphertext_inputs(
        str(function_dir / "program_inputs"), inputs, encryptor_context))
    _, ms["h2d_inputs_ms"] = timed(lambda: functions[name].copy_ciphertext_inputs_to_gpu(
        str(function_dir / "inputs"), 1, encrypted))

    for _ in range(args.warmup):
        program.run()
    runs = [timed(program.run)[1] for _ in range(args.reps)]

    outputs, ms["d2h_outputs_ms"] = timed(functions[name].get_program_outputs)
    actual, ms["decrypt_ms"] = timed(lambda: {
        key: np.asarray(encryptor.decrypt_and_decode(outputs[key], output_scale)) for key in expected})

    model_error = max(float(np.max(np.abs(actual[k] - np.asarray(expected[k])))) for k in expected)
    passed = all(np.allclose(actual[k], expected[k], rtol=1e-5, atol=1e-4) for k in expected)
    iters = int(os.environ.get("CERIUM_RUNTIME_NUM_ITERS", "1"))
    stats = latency_stats(runs)
    compile_record = root / "compile_record.json"
    result = {
        "case": case_description(),
        "graph_launches_per_run": iters,
        "warmup_runs": args.warmup,
        "run_ms": stats,
        "per_launch_median_ms": stats["median"] / iters,
        "per_launch_p95_ms": stats["p95"] / iters,
        "phases_ms": ms,
        "errors": {**errors(actual, enc, runner), "model_max_abs_error": model_error},
        "passed": passed,
        "levels_and_scales": {
            "input_level_limbs": LEVEL, "input_scale_log2": 56,
            "output_level_limbs": LEVEL, "output_scale_log2": int(output_scale).bit_length() - 1,
        },
        "ckks": ckks_parameters(SLOTS, Primes, LEVEL),
        "environment": environment(),
    }
    if compile_record.exists():
        record = json.loads(compile_record.read_text())
        result["compile"] = {k: record[k] for k in record if k != "case"}
    print("\nRESULT " + json.dumps(result), flush=True)
    if not passed:
        raise SystemExit("decrypted output differs from the NumPy model")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--vregs", type=int, default=8192)
    c.add_argument("--bcus", type=int, default=256)
    r = sub.add_parser("run")
    r.add_argument("--warmup", type=int, default=10)
    r.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()
    {"compile": cmd_compile, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
