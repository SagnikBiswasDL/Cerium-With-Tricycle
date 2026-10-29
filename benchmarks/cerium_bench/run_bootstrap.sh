#!/usr/bin/env bash
# Cerium single-GPU bootstrapping benchmark. Usage (on the GPU machine):
#   source /workspace/cerium_work/env.sh
#   bash run_bootstrap.sh [results_dir]
# Compiles benchmarks/bootstrap_32K, then
#   latency: CERIUM_RUNTIME_NUM_ITERS=1,  WARMUP untimed + REPS timed program.run() calls
#   steady:  CERIUM_RUNTIME_NUM_ITERS=10, 1 untimed + 5 timed calls (per launch = run / 10)
# Evaluation keys are several GB; outputs/ is a symlink to SCRATCH (default /root/cerium_scratch; /dev/shm is noexec, so the .so cannot load from it).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${HERE}/results}"
WARMUP="${WARMUP:-3}"
REPS="${REPS:-30}"
SCRATCH="${SCRATCH:-/root/cerium_scratch}"
python="${CERIUM_REPO:?source env.sh first}/venv_cerium/bin/python3"
export SEAL_VERSION="${SEAL_VERSION:-4.1.1}" NCCL_VERSION="${NCCL_VERSION:-2.30.7}"
mkdir -p "${OUT}"
results="${OUT}/bootstrap_results.jsonl"
log="${OUT}/bootstrap.log"
cd "${CERIUM_REPO}/benchmarks/bootstrap_32K"

rm -rf outputs "${SCRATCH}/bootstrap_32K"
mkdir -p "${SCRATCH}/bootstrap_32K"
ln -s "${SCRATCH}/bootstrap_32K" outputs
"${python}" "${HERE}/bench_bootstrap.py" compile 2>&1 | tee -a "${log}" | grep '^COMPILE ' | sed 's/^COMPILE //' > "${OUT}/bootstrap_compile.json"
CERIUM_RUNTIME_NUM_ITERS=1 "${python}" "${HERE}/bench_bootstrap.py" run --warmup "${WARMUP}" --reps "${REPS}" 2>&1 \
    | tee -a "${log}" | grep '^RESULT ' | sed 's/^RESULT //' >> "${results}"
CERIUM_RUNTIME_NUM_ITERS=10 "${python}" "${HERE}/bench_bootstrap.py" run --warmup 1 --reps 5 2>&1 \
    | tee -a "${log}" | grep '^RESULT ' | sed 's/^RESULT //' >> "${results}"
echo "Done: ${results}" | tee -a "${log}"
