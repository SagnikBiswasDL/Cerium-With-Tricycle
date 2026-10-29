"""Benchmark Cerium's single-GPU CKKS bootstrapping (benchmarks/bootstrap_32K).

Run from Cerium/benchmarks/bootstrap_32K with the Cerium virtualenv and
PYTHONPATH (env.sh); outputs/ may be a symlink to scratch space (the
evaluation keys are several GB).

    python bench_bootstrap.py compile [--vregs 4096 --bcus 128]
    python bench_bootstrap.py run [--warmup 3 --reps 30]

The program is bootstrap_32K/bootstrap.py unchanged (function
bootstrap_32Kslots_33levels_1gpus); the run step reproduces
bootstrap_runner.py (same seeds, keys and input) with per-phase timing.
Phases are defined as in bench_bmm.py. The input is 32768 complex numbers of
modulus 1 with uniform random phase, encrypted at 2 limbs.

Errors are measured against the original plaintext vector m:
    before  = decrypt(input ciphertext) - m      (encoding + encryption error)
    after   = decrypt(bootstrapped output) - m   (total error after bootstrapping)
    added   = decrypt(output) - decrypt(input)   (error added by bootstrapping)
For each: max_abs, mean_abs, rel_l2 = ||e||_2 / ||m||_2, and precision bits
= -log2(max_abs).
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_info import ckks_parameters, environment  # noqa: E402
from opcount import count_compiled, count_source  # noqa: E402
from bench_bmm import latency_stats, timed  # noqa: E402

FUNCTION = "bootstrap_32Kslots_33levels_1gpus"
SLOTS = 32 * 1024
INPUT_LEVEL = 2


def cmd_compile(args):
    sys.path.insert(0, os.getcwd())
    import bootstrap as program_module
    from cerium.compiler import cerium_compile

    record = {"vregs": args.vregs, "bcus": args.bcus}
    original_output = program_module.Output

    def output(name, expr):
        record["output"] = {"name": name, "level_limbs": expr.level(), "scale_log2": expr.scale()}
        return original_output(name, expr)
    program_module.Output = output

    start = time.perf_counter()
    with count_source() as tracker:
        program, record["dsl_build_ms"] = timed(program_module.bootstrap_single_gpu)
    record["source_ops"] = dict(tracker.counts)

    os.environ["CERIUM_PRINT_GRAPHS"] = "1"
    _, record["cerium_compile_ms"] = timed(lambda: cerium_compile(program, 1, args.vregs, args.bcus, "outputs/"))
    graphs = Path("outputs/graphs")
    graphs.mkdir(parents=True, exist_ok=True)
    for dot in Path(".").glob(f"{FUNCTION}_cerium_graph*"):
        shutil.move(str(dot), graphs / dot.name)

    build, record["nvcc_build_ms"] = timed(lambda: subprocess.run(
        ["make", "all", "-j", f"FNAME={FUNCTION}", "DIRNAME=outputs"], capture_output=True, text=True))
    if build.returncode != 0:
        sys.stderr.write(build.stdout[-4000:] + build.stderr[-4000:])
        raise SystemExit("make failed")
    compiled = Path("outputs/compiled") / FUNCTION
    compiled.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path("outputs") / FUNCTION / "output", compiled, dirs_exist_ok=True)
    record["compile_total_ms"] = (time.perf_counter() - start) * 1e3
    record["compiled_ops"] = count_compiled(
        graphs / f"{FUNCTION}_cerium_graph_simple.dot",
        Path("outputs") / FUNCTION / "partition_0" / "ll.functions")
    Path("outputs/compile_record.json").write_text(json.dumps(record, indent=1))
    print("\nCOMPILE " + json.dumps(record), flush=True)


def _size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def error_stats(error, reference):
    max_abs = float(np.max(np.abs(error)))
    return {
        "max_abs": max_abs,
        "mean_abs": float(np.mean(np.abs(error))),
        "rel_l2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
        "precision_bits": float(-np.log2(max_abs)),
        "max_abs_real": float(np.max(np.abs(error.real))),
        "max_abs_imag": float(np.max(np.abs(error.imag))),
    }


def cmd_run(args):
    sys.path.insert(0, os.getcwd())
    import cerium.runtime as runtime
    from bootstrap_values import get_bootstrap_runner_inputs
    from primes import Primes
    from bootstrap_runner import generate_secret_key

    # Same seeds and call order as bootstrap_runner.run_bootstrap.
    random.seed(10)
    np.random.seed(10)
    ms = {}
    raw_inputs, ms["bootstrap_constants_ms"] = timed(lambda: get_bootstrap_runner_inputs("b33", 18))

    def setup_context():
        secret = generate_secret_key(SLOTS, HammingWeight=SLOTS)
        ephemeral = generate_secret_key(SLOTS, HammingWeight=32)
        seed = np.random.randint(0, 2**32 - 1, size=8).tolist()
        context = runtime.Context(SLOTS, Primes)
        encryptor_context = runtime.CKKSEncryptorContext(context, secret, ephemeral, seed)
        return context, encryptor_context, runtime.CKKSEncryptor(encryptor_context, seed)

    (context, encryptor_context, encryptor), ms["keygen_secret_ms"] = timed(setup_context)
    log_message_ratio = 5
    input_scale = Primes[1] * Primes[0] / (1 << log_message_ratio)
    message = np.array([np.exp(1j * random.uniform(0, 2 * np.pi)) for _ in range(SLOTS)], dtype=np.complex128)

    base = "outputs/compiled"
    constants = runtime.RawInputsWrapper(raw_inputs)
    raw_inputs.clear()
    generator = runtime.IOGenerator(context)
    _, ms["keygen_eval_ms"] = timed(lambda: generator.generate_and_serialize_evalkeys(
        f"{base}/{FUNCTION}/evalkeys", f"{base}/{FUNCTION}/program_inputs", encryptor_context))
    _, ms["encode_plaintexts_ms"] = timed(lambda: generator.generate_and_serialize_plaintexts(
        f"{base}/{FUNCTION}/plaintexts", f"{base}/{FUNCTION}/program_inputs", constants))
    key_bytes = _size(Path(f"{base}/{FUNCTION}/evalkeys"))
    plaintext_bytes = _size(Path(f"{base}/{FUNCTION}/plaintexts"))

    program = runtime.FusedKernelProgram(context, base, FUNCTION)
    functions, ms["load_program_ms"] = timed(
        lambda: program.make_program(1, {FUNCTION: {"use_cudagraph": "True"}}, constants))
    function = functions[FUNCTION]
    inputs = runtime.RawInputsWrapper({"x": (message, input_scale)})
    encrypted, ms["encrypt_ms"] = timed(lambda: generator.generate_ciphertext_inputs(
        f"{base}/{FUNCTION}/program_inputs", inputs, encryptor_context))
    _, ms["h2d_inputs_ms"] = timed(lambda: function.copy_ciphertext_inputs_to_gpu(
        f"{base}/{FUNCTION}/inputs", 1, encrypted))

    before_decrypted = np.array(encryptor.decrypt_and_decode(encrypted["x"], input_scale))

    for _ in range(args.warmup):
        program.run()
    runs = [timed(program.run)[1] for _ in range(args.reps)]

    outputs, ms["d2h_outputs_ms"] = timed(function.get_program_outputs)
    after_decrypted, ms["decrypt_ms"] = timed(
        lambda: np.array(encryptor.decrypt_and_decode(outputs["z"], input_scale)))

    iters = int(os.environ.get("CERIUM_RUNTIME_NUM_ITERS", "1"))
    stats = latency_stats(runs)
    record = json.loads(Path("outputs/compile_record.json").read_text()) \
        if Path("outputs/compile_record.json").exists() else {}
    result = {
        "program": f"Cerium benchmarks/bootstrap_32K {FUNCTION}",
        "graph_launches_per_run": iters,
        "warmup_runs": args.warmup,
        "run_ms": stats,
        "per_launch_median_ms": stats["median"] / iters,
        "per_launch_p95_ms": stats["p95"] / iters,
        "phases_ms": ms,
        "eval_key_bytes": key_bytes,
        "plaintext_bytes": plaintext_bytes,
        "errors": {
            "before_bootstrap_vs_plaintext": error_stats(before_decrypted - message, message),
            "after_bootstrap_vs_plaintext": error_stats(after_decrypted - message, message),
            "added_by_bootstrap": error_stats(after_decrypted - before_decrypted, message),
        },
        "levels_and_scales": {
            "input_level_limbs": INPUT_LEVEL,
            "input_scale_log2_declared": 56,
            "input_scale_log2_encoded": float(np.log2(input_scale)),
            "mod_raise_to_limbs": 18 + 33,
            "output": record.get("output"),
            "output_decode_scale_log2": float(np.log2(input_scale)),
            "secret_key_hamming_weight": SLOTS,
            "ephemeral_key_hamming_weight": 32,
            "homomorphic_mod": "Chebyshev degree 30 + 3 double-angle squarings (HK_deg=30, HK_r=3)",
        },
        "ckks": ckks_parameters(SLOTS, Primes, INPUT_LEVEL),
        "environment": environment(),
        "compile": record,
    }
    print("\nRESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--vregs", type=int, default=4096)
    c.add_argument("--bcus", type=int, default=128)
    r = sub.add_parser("run")
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()
    {"compile": cmd_compile, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
