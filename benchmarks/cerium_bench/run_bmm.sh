#!/usr/bin/env bash
# BMM benchmark sweep on one GPU. Usage (on the GPU machine):
#   source /workspace/cerium_work/env.sh
#   bash run_bmm.sh [results_dir]
# Each case: compile (timed, op counts), then
#   latency: CERIUM_RUNTIME_NUM_ITERS=1,   WARMUP untimed + REPS timed program.run() calls
#   steady:  CERIUM_RUNTIME_NUM_ITERS=100, 2 untimed + 20 timed calls (per launch = run / 100)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${HERE}/results}"
WARMUP="${WARMUP:-10}"
REPS="${REPS:-200}"
SCRATCH="${SCRATCH:-/root/cerium_scratch}"
EXAMPLES="${CERIUM_REPO:?source env.sh first}/examples"
python="${CERIUM_REPO}/venv_cerium/bin/python3"
export SEAL_VERSION="${SEAL_VERSION:-4.1.1}" NCCL_VERSION="${NCCL_VERSION:-2.30.7}"
mkdir -p "${OUT}"
results="${OUT}/bmm_results.jsonl"
log="${OUT}/bmm_sweep.log"
cd "${EXAMPLES}"

measure() {  # env assignments for one case
    echo "== $*" | tee -a "${log}"
    rm -rf bicyclic_matmul/outputs "${SCRATCH}/bicyclic_matmul"
    mkdir -p "${SCRATCH}/bicyclic_matmul"
    ln -s "${SCRATCH}/bicyclic_matmul" bicyclic_matmul/outputs
    env "$@" "${python}" "${HERE}/bench_bmm.py" compile >> "${log}" 2>&1 || { echo "COMPILE FAILED $*" | tee -a "${log}"; return 0; }
    env "$@" CERIUM_RUNTIME_NUM_ITERS=1 "${python}" "${HERE}/bench_bmm.py" run --warmup "${WARMUP}" --reps "${REPS}" 2>&1 \
        | tee -a "${log}" | grep '^RESULT ' | sed 's/^RESULT //' >> "${results}" || echo "RUN FAILED $*" | tee -a "${log}"
    env "$@" CERIUM_RUNTIME_NUM_ITERS=100 "${python}" "${HERE}/bench_bmm.py" run --warmup 2 --reps 20 2>&1 \
        | tee -a "${log}" | grep '^RESULT ' | sed 's/^RESULT //' >> "${results}" || echo "RUN FAILED $*" | tee -a "${log}"
}

CASES="${CASES:-all}"
if [[ "${CASES}" == tiny || "${CASES}" == all ]]; then
    # Correctness check: the original tiny example (integer entries).
    measure BICYCLIC_SHAPE=3,4,5 BICYCLIC_SCHEDULE=linear BICYCLIC_VALUES=int
fi
if [[ "${CASES}" == all ]]; then
    # One product that nearly fills the 32768 slots (M*N*P close to 32768).
    measure BICYCLIC_SHAPE=31,32,33  BICYCLIC_SCHEDULE=doubling BICYCLIC_VALUES=real
    measure BICYCLIC_SHAPE=31,32,33  BICYCLIC_SCHEDULE=linear   BICYCLIC_VALUES=real
    measure BICYCLIC_SHAPE=31,32,33  BICYCLIC_SCHEDULE=hoisted  BICYCLIC_VALUES=real
    measure BICYCLIC_SHAPE=31,32,33  BICYCLIC_SCHEDULE=rotacc   BICYCLIC_VALUES=real
    measure BICYCLIC_SHAPE=15,128,17 BICYCLIC_SCHEDULE=doubling BICYCLIC_VALUES=real
    measure BICYCLIC_SHAPE=17,64,29  BICYCLIC_SCHEDULE=doubling BICYCLIC_VALUES=real
    measure BICYCLIC_SHAPE=127,16,15 BICYCLIC_SCHEDULE=doubling BICYCLIC_VALUES=real
    measure BICYCLIC_SHAPE=127,16,15 BICYCLIC_SCHEDULE=linear   BICYCLIC_VALUES=real
    # Slot batching: many independent small products in one ciphertext pair.
    measure BICYCLIC_SHAPE=3,4,5    BICYCLIC_SCHEDULE=doubling BICYCLIC_SLOT_BATCH=max BICYCLIC_VALUES=real
    measure BICYCLIC_SHAPE=7,8,9    BICYCLIC_SCHEDULE=doubling BICYCLIC_SLOT_BATCH=max BICYCLIC_VALUES=real
    measure BICYCLIC_SHAPE=15,16,17 BICYCLIC_SCHEDULE=doubling BICYCLIC_SLOT_BATCH=max BICYCLIC_VALUES=real
    # Ablations for time attribution (same ciphertext size; shape does not matter).
    measure BICYCLIC_SHAPE=31,32,33 BICYCLIC_SCHEDULE=none BICYCLIC_VALUES=real
    measure BICYCLIC_SHAPE=31,32,33 BICYCLIC_SCHEDULE=add  BICYCLIC_VALUES=real
fi
echo "Done: ${results}" | tee -a "${log}"
