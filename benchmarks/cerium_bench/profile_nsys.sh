#!/usr/bin/env bash
# GPU kernel-time breakdown with Nsight Systems. Usage (on the GPU machine):
#   source /workspace/cerium_work/env.sh
#   bash profile_nsys.sh [results_dir]
# Profiles bench_bmm.py / bench_bootstrap.py `run` (CERIUM_RUNTIME_NUM_ITERS=1)
# with --cuda-graph-trace=node so each kernel inside the CUDA graph is timed,
# then writes <name>_kern_sum.csv and <name>_breakdown.json (per-launch times).
# Run bootstrap after run_bootstrap.sh so its compiled program is in place.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${HERE}/results}/nsys"
SCRATCH="${SCRATCH:-/root/cerium_scratch}"
WARMUP=2
REPS=20
python="${CERIUM_REPO:?source env.sh first}/venv_cerium/bin/python3"
mkdir -p "${OUT}"
log="${OUT}/profile.log"

profile() {  # name, then the command
    local name="$1"; shift
    CERIUM_RUNTIME_NUM_ITERS=1 nsys profile --cuda-graph-trace=node -t cuda -f true -o "${OUT}/${name}" "$@" >> "${log}" 2>&1
    nsys stats -r cuda_gpu_kern_sum -f csv -o "${OUT}/${name}" "${OUT}/${name}.nsys-rep" >> "${log}" 2>&1
    "${python}" "${HERE}/kernel_breakdown.py" "${OUT}/${name}_cuda_gpu_kern_sum.csv" $((WARMUP + REPS)) > "${OUT}/${name}_breakdown.json"
    rm -f "${OUT}/${name}.sqlite"
    echo "${name}: $(grep kernel_time_per_launch_ms "${OUT}/${name}_breakdown.json")" | tee -a "${log}"
}

cd "${CERIUM_REPO}/examples"
for schedule in none linear doubling hoisted rotacc; do
    export BICYCLIC_SHAPE=31,32,33 BICYCLIC_SCHEDULE="${schedule}" BICYCLIC_VALUES=real
    rm -rf bicyclic_matmul/outputs "${SCRATCH}/bicyclic_matmul"
    mkdir -p "${SCRATCH}/bicyclic_matmul"
    ln -s "${SCRATCH}/bicyclic_matmul" bicyclic_matmul/outputs
    "${python}" "${HERE}/bench_bmm.py" compile >> "${log}" 2>&1
    profile "bmm_31x32x33_${schedule}" "${python}" "${HERE}/bench_bmm.py" run --warmup "${WARMUP}" --reps "${REPS}"
done
unset BICYCLIC_SHAPE BICYCLIC_SCHEDULE BICYCLIC_VALUES

cd "${CERIUM_REPO}/benchmarks/bootstrap_32K"
profile bootstrap "${python}" "${HERE}/bench_bootstrap.py" run --warmup "${WARMUP}" --reps "${REPS}"
echo "Done: ${OUT}" | tee -a "${log}"
