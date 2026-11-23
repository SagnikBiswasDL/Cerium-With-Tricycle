#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One unattended pass on a fresh Runpod Pod: build Cerium (setup_pod.sh) at the
# commit this repository was written against, then compile and run each shape.
# Run it under nohup; progress goes to ${CERIUM_WORK}/bootstrap.log.
#
# Usage: bash bootstrap_pod.sh [SHAPES...]     default: mini tiny
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK="${CERIUM_WORK:-/workspace/cerium_work}"
CERIUM_COMMIT="${CERIUM_COMMIT:-6b48704}"
shapes=("$@")
(( ${#shapes[@]} )) || shapes=(mini tiny)

# Some data centers time out on plain-HTTP archive.ubuntu.com.
sed -i 's#http://archive.ubuntu.com/ubuntu/#https://archive.ubuntu.com/ubuntu/#' \
    /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true

mkdir -p "${WORK}"
if [[ ! -d "${WORK}/Cerium/.git" ]]; then
    git clone -q --recurse-submodules https://github.com/CMU-CAOS/Cerium.git "${WORK}/Cerium"
fi
git -C "${WORK}/Cerium" checkout -q "${CERIUM_COMMIT}"
git -C "${WORK}/Cerium" submodule update -q --init --recursive

JOBS="${JOBS:-$(nproc)}" bash "${ROOT}/familiarization/pod/setup_pod.sh" "${CUDA_ARCH:-90}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

for shape in "${shapes[@]}"; do
    bash "${SCRIPT_DIR}/run_pod.sh" "${shape}" 1 || echo "== ${shape} FAILED"
done
echo "== bootstrap done"
