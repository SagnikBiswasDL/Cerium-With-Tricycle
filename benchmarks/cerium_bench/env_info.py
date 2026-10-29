"""Hardware, software and CKKS parameter record attached to every result."""

import math
import os
import platform
import subprocess
import sys


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=True).stdout.strip()
    except Exception as e:  # noqa: BLE001 - recorded, not fatal
        return f"unavailable: {e}"


def environment():
    import numpy as np

    repo = os.environ.get("CERIUM_REPO", "")
    commit = _run(f"git -C '{repo}' rev-parse HEAD 2>/dev/null") if repo else ""
    return {
        "gpu": _run("nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv,noheader"),
        "gpu_count": _run("nvidia-smi -L | wc -l"),
        # Other GPU tenants make latency numbers unrepresentative; recorded at the end of the run.
        "gpu_compute_processes": _run("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"),
        "gpu_utilization_sample": _run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"),
        "cuda_driver": _run("nvidia-smi | sed -n 's/.*CUDA Version: \\([0-9.]*\\).*/\\1/p'"),
        # The example and benchmark Makefiles hard-code /usr/local/cuda-13.3/bin/nvcc.
        "nvcc": _run(f"{os.environ.get('CERIUM_NVCC', '/usr/local/cuda-13.3/bin/nvcc')} --version 2>/dev/null"
                     " | grep -o 'release [0-9.]*'"),
        "cpu": _run("lscpu | sed -n 's/^Model name: *//p'"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cerium_commit": commit or os.environ.get("CERIUM_COMMIT", "unknown"),
        "seal": os.environ.get("SEAL_VERSION", "unknown"),
        "nccl": os.environ.get("NCCL_VERSION", "unknown"),
        "cerium_env": {k: v for k, v in os.environ.items() if k.startswith("CERIUM_") and k != "CERIUM_REPO"},
    }


def ckks_parameters(slots, primes, level):
    """level = number of RNS limbs the program's input ciphertexts carry."""
    used = primes[:level]
    return {
        "ring_dimension": 2 * slots,
        "slots": slots,
        "rns_prime_bits": [round(math.log2(p), 3) for p in used],
        "input_level_limbs": level,
        "log2_Q_at_input_level": round(sum(math.log2(p) for p in used), 2),
        "primes_in_table": len(primes),
    }
