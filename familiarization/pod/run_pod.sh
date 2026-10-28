#!/usr/bin/env bash
# Native (no Docker) equivalent of examples/compile.sh + examples/run_example.sh.
#
# Usage: bash run_pod.sh EXAMPLE [GPUS]     e.g. bash run_pod.sh addition 1
set -euo pipefail

WORK="${CERIUM_WORK:-/workspace/cerium_work}"
source "${WORK}/env.sh"
example="${1:?Usage: run_pod.sh EXAMPLE [GPUS]}"
gpus="${2:-1}"
examples_dir="${CERIUM_REPO}/examples"
python="${CERIUM_REPO}/venv_cerium/bin/python3"
log_dir="${WORK}/logs"
mkdir -p "${log_dir}"

cd "${examples_dir}"
echo "== Compiling ${example} for ${gpus} GPU(s)"
"${python}" -m "examples.${example}.program" --gpus "${gpus}" 2>&1 | tee "${log_dir}/${example}_compile_${gpus}gpus.log"
bash "${examples_dir}/make_example.sh" "${example}" "${gpus}" >> "${log_dir}/${example}_compile_${gpus}gpus.log" 2>&1

echo "== Running ${example} on ${gpus} GPU(s)"
"${python}" -m "examples.${example}.runner" --gpus "${gpus}" 2>&1 | tee "${log_dir}/${example}_run_${gpus}gpus.log"
