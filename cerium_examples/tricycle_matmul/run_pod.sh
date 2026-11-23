#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Compile and run the Tricycle matmul example natively on a Runpod Pod set up
# with familiarization/pod/setup_pod.sh (no Docker).
#
# Usage: bash run_pod.sh SHAPE [GPUS]     SHAPE is mini, tiny or base
# Environment: TRICYCLE_MATMULS, CERIUM_VREGS, CERIUM_BCUS, MAKE_JOBS,
# CERIUM_WORK (default /workspace/cerium_work).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK="${CERIUM_WORK:-/workspace/cerium_work}"
source "${WORK}/env.sh"
export TRICYCLE_SHAPE="${1:?Usage: run_pod.sh mini|tiny|base [GPUS]}"
gpus="${2:-1}"
case "${TRICYCLE_SHAPE}" in
    mini) : "${CERIUM_VREGS:=8192}" "${CERIUM_BCUS:=256}" ;;
    tiny) : "${CERIUM_VREGS:=16384}" "${CERIUM_BCUS:=1024}" ;;
    base) : "${CERIUM_VREGS:=32768}" "${CERIUM_BCUS:=2048}" ;;
    *) echo "SHAPE must be mini, tiny or base" >&2; exit 2 ;;
esac
export CERIUM_VREGS CERIUM_BCUS
export PYTHONPATH="${PYTHONPATH}:${ROOT}"

examples="${CERIUM_REPO}/examples"
python="${CERIUM_REPO}/venv_cerium/bin/python3"
tag="${TRICYCLE_SHAPE}${TRICYCLE_MATMULS:+_${TRICYCLE_MATMULS//,/-}}_${gpus}gpus"
logs="${WORK}/logs"
out="${examples}/tricycle_matmul/outputs/tricycle_matmul"
mkdir -p "${logs}" "${examples}/tricycle_matmul"
cp "${SCRIPT_DIR}"/*.py "${examples}/tricycle_matmul/"
rm -rf "${out}"

cd "${examples}"
echo "== Compiling ${tag} (vregs ${CERIUM_VREGS}, bcus ${CERIUM_BCUS})"
start=$(date +%s)
"${python}" -m examples.tricycle_matmul.program --gpus "${gpus}" 2>&1 | tee "${logs}/tricycle_${tag}_compile.log"
make -C "${examples}" -j"${MAKE_JOBS:-16}" all "FNAME=main_${gpus}gpus" \
    "DIRNAME=tricycle_matmul/outputs/tricycle_matmul" >> "${logs}/tricycle_${tag}_compile.log" 2>&1
mkdir -p "${out}/compiled/main_${gpus}gpus"
cp -a "${out}/main_${gpus}gpus/output/." "${out}/compiled/main_${gpus}gpus/"
echo "== Compiled in $(( $(date +%s) - start ))s"

echo "== Running ${tag}"
start=$(date +%s)
"${python}" -m examples.tricycle_matmul.runner --gpus "${gpus}" 2>&1 | tee "${logs}/tricycle_${tag}_run.log"
echo "== Ran in $(( $(date +%s) - start ))s"
