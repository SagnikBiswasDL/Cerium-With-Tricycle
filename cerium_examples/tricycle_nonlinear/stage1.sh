#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Stage 1 of the full-layer measurements on one GPU: Cerium's bootstrap cost,
# then GELU at the two scale settings (2 or 3 primes per multiplicative level).
# Waits for setup_pod.sh to finish first.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK="${CERIUM_WORK:-/workspace/cerium_work}"
until [[ -x /usr/local/cuda-13.3/bin/nvcc ]] && ! pgrep -f setup_pod.sh >/dev/null; do sleep 20; done
source "${WORK}/env.sh"

if ! command -v nsys >/dev/null && ! ls /usr/local/cuda*/bin/nsys >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nsight-systems-13-3 >/dev/null 2>&1 \
        || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nsight-systems >/dev/null 2>&1
fi

echo "== bootstrap $(date)"
bash "${ROOT}/benchmarks/cerium_bench/run_bootstrap.sh" "${WORK}/logs/bootstrap_h100"

for limbs in 2 3; do
    level=$(( limbs == 2 ? 21 : 30 ))
    echo "== gelu limbs ${limbs} $(date)"
    TRICYCLE_NONLINEAR="gelu" TRICYCLE_LIMBS="${limbs}" TRICYCLE_INPUT_LEVEL="${level}" \
        TRICYCLE_PROFILE=1 bash "${SCRIPT_DIR}/run_pod.sh" "gelu_l${limbs}"
done
echo "== stage1 done $(date)"
