#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Emit, compile or run the Tricycle matmul example in Cerium's Docker image.
# Cerium's compile.sh mounts only the Cerium checkout; this example also needs
# the tricycle package next to it, so the whole repository is mounted.
#
# Usage: tricycle.sh emit|compile|run [1|2|4|8]
# Environment: TRICYCLE_SHAPE (mini|tiny|base), TRICYCLE_MATMULS, CERIUM_VREGS,
# CERIUM_BCUS, MAKE_JOBS (cap on parallel nvcc jobs), CERIUM_ROOT (default:
# <repo>/Cerium), CERIUM_IMAGE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CERIUM="${CERIUM_ROOT:-${ROOT}/Cerium}"
EXAMPLES="${CERIUM}/examples"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
PYTHON="${CERIUM}/venv_cerium/bin/python3"
mode="${1:?Usage: tricycle.sh emit|compile|run [GPUS]}"
gpus="${2:-1}"

mkdir -p "${EXAMPLES}/tricycle_matmul"
cp "${SCRIPT_DIR}"/*.py "${EXAMPLES}/tricycle_matmul/"

docker_args=(--rm --user "$(id -u):$(id -g)"
    -v "${ROOT}:${ROOT}" -w "${EXAMPLES}"
    -e "PYTHONPATH=${CERIUM}:${CERIUM}/build/cerium/python:${ROOT}"
    -e "LD_LIBRARY_PATH=${CERIUM}/build/cerium/runtime"
    -e TRICYCLE_SHAPE -e TRICYCLE_MATMULS -e TRICYCLE_LEVELS -e CERIUM_VREGS -e CERIUM_BCUS)
if [[ "${CERIUM}" != "${ROOT}"/* ]]; then
    docker_args+=(-v "${CERIUM}:${CERIUM}")
fi

case "${mode}" in
    emit)
        docker run "${docker_args[@]}" "${IMAGE_NAME}" \
            "${PYTHON}" -m examples.tricycle_matmul.program --gpus "${gpus}" --emit-only ;;
    compile)
        docker run "${docker_args[@]}" "${IMAGE_NAME}" \
            "${PYTHON}" -m examples.tricycle_matmul.program --gpus "${gpus}"
        if [[ -z "${MAKE_JOBS:-}" ]]; then
            docker run "${docker_args[@]}" "${IMAGE_NAME}" \
                bash "${EXAMPLES}/make_example.sh" tricycle_matmul "${gpus}"
        else
            # make_example.sh runs an unbounded make -j; BERT-Base's nvcc jobs
            # exhaust an 8 GB Docker VM that way.
            out="${EXAMPLES}/tricycle_matmul/outputs/tricycle_matmul"
            docker run "${docker_args[@]}" "${IMAGE_NAME}" bash -c "
                make -C '${EXAMPLES}' -j${MAKE_JOBS} all FNAME=main_${gpus}gpus \
                    DIRNAME=tricycle_matmul/outputs/tricycle_matmul &&
                mkdir -p '${out}/compiled/main_${gpus}gpus' &&
                cp -a '${out}/main_${gpus}gpus/output/.' '${out}/compiled/main_${gpus}gpus/'"
        fi ;;
    run)
        docker run --gpus all "${docker_args[@]}" "${IMAGE_NAME}" \
            "${PYTHON}" -m examples.tricycle_matmul.runner --gpus "${gpus}" ;;
    *)
        echo "mode must be emit, compile or run" >&2
        exit 2 ;;
esac
