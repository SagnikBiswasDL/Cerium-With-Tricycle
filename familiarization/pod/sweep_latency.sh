#!/usr/bin/env bash
# Latency sweep for the matrix examples on 1 GPU. For each configuration:
# clean compile (timed), then bench_matmul.py twice:
#   single: 50 separate program.run() calls, 1 CUDA-graph launch each
#   steady: 5 program.run() calls, 100 back-to-back launches each
# Writes RESULT JSON lines to $WORK/logs/latency_results.jsonl.
# SWEEP=bert instead times BERT's 128x128 attention matmul with the compiler
# settings bert_cerium.py uses for it (3096 vregs, 128 bcus, block size 256).
set -euo pipefail

WORK="${CERIUM_WORK:-/workspace/cerium_work}"
source "${WORK}/env.sh"
python="${CERIUM_REPO}/venv_cerium/bin/python3"
SWEEP="${SWEEP:-matrix}"
suffix=$([[ "${SWEEP}" == bert ]] && echo "_bert" || true)
results="${WORK}/logs/latency_results${suffix}.jsonl"
# Compiler resources for every configuration. examples/common.py uses 1024
# vregs and 128 bcus, which is too few for the largest shapes; Cerium's own
# benchmarks use 4096-8192 vregs.
if [[ "${SWEEP}" == bert ]]; then
    VREGS="${VREGS:-3096}"
    BCUS="${BCUS:-128}"
else
    VREGS="${VREGS:-8192}"
    BCUS="${BCUS:-256}"
fi
log="${WORK}/logs/latency_sweep${suffix}.log"
# Compiled outputs (mostly evaluation keys, up to several GB) go on the
# container disk: /workspace is a network volume with a tight quota.
SCRATCH="${CERIUM_SCRATCH:-/root/cerium_scratch}"
cd "${CERIUM_REPO}/examples"
: > "${results}"

measure() {  # example, then env assignments
    local example="$1"; shift
    echo "== ${example} $*" | tee -a "${log}"
    rm -rf "${example}/outputs" "${SCRATCH:?}/${example}"
    mkdir -p "${SCRATCH}/${example}"
    ln -s "${SCRATCH}/${example}" "${example}/outputs"
    local start end
    start=$(date +%s.%N)
    env "$@" "${python}" -m "examples.${example}.program" --gpus 1 --vregs "${VREGS}" --bcus "${BCUS}" >> "${log}" 2>&1
    bash make_example.sh "${example}" 1 >> "${log}" 2>&1
    end=$(date +%s.%N)
    echo "COMPILE ${example} $* $(python3 -c "print(round((${end} - ${start}) * 1000))") ms" | tee -a "${log}" "${results}.compile"
    env "$@" CERIUM_RUNTIME_NUM_ITERS=1 "${python}" "${WORK}/bench_matmul.py" "${example}" --calls 50 2>&1 \
        | tee -a "${log}" | grep '^RESULT ' | sed 's/^RESULT //' >> "${results}"
    env "$@" CERIUM_RUNTIME_NUM_ITERS=100 "${python}" "${WORK}/bench_matmul.py" "${example}" --calls 5 2>&1 \
        | tee -a "${log}" | grep '^RESULT ' | sed 's/^RESULT //' >> "${results}"
}

: > "${results}.compile"
if [[ "${SWEEP}" == bert ]]; then
    measure bert_attention_matmul CERIUM_BLOCK_SIZE=256
    echo "Done: ${results}"
    exit 0
fi
for shape in 3,4,5 7,8,9 15,16,17 31,32,33; do
    for schedule in linear doubling; do
        measure bicyclic_matmul "BICYCLIC_SHAPE=${shape}" "BICYCLIC_SCHEDULE=${schedule}"
    done
done
for size in 16 32 64 128; do
    measure diagonal_matvec "DIAGONAL_SIZE=${size}"
done
echo "Done: ${results}"
