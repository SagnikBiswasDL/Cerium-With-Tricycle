#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Compile and run the Tricycle nonlinear kernels natively on a Runpod Pod set up
# with familiarization/pod/setup_pod.sh (no Docker).
#
# Usage: bash run_pod.sh TAG     with TRICYCLE_NONLINEAR, TRICYCLE_LIMBS,
# TRICYCLE_INPUT_LEVEL, CERIUM_VREGS, CERIUM_BCUS and TRICYCLE_PROFILE in the
# environment (see program.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK="${CERIUM_WORK:-/workspace/cerium_work}"
source "${WORK}/env.sh"
tag="${1:?Usage: run_pod.sh TAG}"
gpus=1
: "${CERIUM_VREGS:=8192}" "${CERIUM_BCUS:=128}"
export CERIUM_VREGS CERIUM_BCUS
export PYTHONPATH="${PYTHONPATH}:${ROOT}"

examples="${CERIUM_REPO}/examples"
python="${CERIUM_REPO}/venv_cerium/bin/python3"
logs="${WORK}/logs/nonlinear"
out="${examples}/tricycle_nonlinear/outputs/tricycle_nonlinear"
mkdir -p "${logs}" "${examples}/tricycle_nonlinear"
cp "${SCRIPT_DIR}"/*.py "${examples}/tricycle_nonlinear/"
rm -rf "${out}"

cd "${examples}"
echo "== Compiling ${tag}: ${TRICYCLE_NONLINEAR:-gelu} (limbs ${TRICYCLE_LIMBS:-2}, level ${TRICYCLE_INPUT_LEVEL:-21}, vregs ${CERIUM_VREGS}, bcus ${CERIUM_BCUS})"
start=$(date +%s)
"${python}" -m examples.tricycle_nonlinear.program --gpus "${gpus}" 2>&1 | tee "${logs}/${tag}_compile.log"
# Every compiled function: main and, for fused kernels, Cerium BERT's bootstrap.
for fdir in "${out}"/*/; do
    fname="$(basename "${fdir}")"
    [[ "${fname}" == compiled || "${fname}" == graphs ]] && continue
    make -C "${examples}" -j"${MAKE_JOBS:-16}" all "FNAME=${fname}" \
        "DIRNAME=tricycle_nonlinear/outputs/tricycle_nonlinear" >> "${logs}/${tag}_compile.log" 2>&1
    mkdir -p "${out}/compiled/${fname}"
    cp -r "${fdir}/output/." "${out}/compiled/${fname}/"
done
echo "== Compiled in $(( $(date +%s) - start ))s"

echo "== Running ${tag}"
"${python}" -m examples.tricycle_nonlinear.runner --gpus "${gpus}" 2>&1 | tee "${logs}/${tag}_run.log"

if [[ -n "${TRICYCLE_PROFILE:-}" ]]; then
    nsys="$(command -v nsys || ls /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)"
    warmup=2 reps=5
    echo "== Profiling ${tag} with ${nsys}"
    TRICYCLE_WARMUP="${warmup}" TRICYCLE_REPS="${reps}" "${nsys}" profile --trace=cuda \
        --cuda-graph-trace=node --force-overwrite=true --output="${logs}/${tag}" \
        "${python}" -m examples.tricycle_nonlinear.runner --gpus "${gpus}" > /dev/null 2>&1
    "${nsys}" stats --report cuda_gpu_kern_sum --format csv --force-export=true \
        --output="${logs}/${tag}" "${logs}/${tag}.nsys-rep" > /dev/null
    rm -f "${logs}/${tag}.sqlite" "${logs}/${tag}.nsys-rep"
    "${python}" "${ROOT}/benchmarks/cerium_bench/kernel_breakdown.py" \
        "${logs}/${tag}_cuda_gpu_kern_sum.csv" $((1 + warmup + reps)) \
        | tee "${logs}/${tag}_breakdown.json"
fi
