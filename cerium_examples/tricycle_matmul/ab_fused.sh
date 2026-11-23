#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Current versus fused (BsgsMultiplyAccumulate) plaintext products on BERT-Base,
# one split program at a time, each with warm timing and an nsys kernel
# breakdown. Builds Cerium first if needed. Run it under nohup.
#
# Usage: bash ab_fused.sh [GPUS]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${CERIUM_WORK:-/workspace/cerium_work}"
gpus="${1:-1}"

bash "${SCRIPT_DIR}/bootstrap_pod.sh" mini
source "${WORK}/env.sh"
mkdir -p "${WORK}/logs/mini_current"
mv "${WORK}"/logs/tricycle_mini_*gpus_*.log "${WORK}/logs/mini_current/" 2>/dev/null
if ! command -v nsys >/dev/null && ! ls /usr/local/cuda*/bin/nsys >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nsight-systems-13-3 >/dev/null 2>&1 \
        || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nsight-systems >/dev/null 2>&1
fi
profile=1
command -v nsys >/dev/null || ls /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys \
    >/dev/null 2>&1 || { echo "== nsys not found, timing only"; profile=""; }

echo "== mini, fused"
TRICYCLE_FUSED=all bash "${SCRIPT_DIR}/run_pod.sh" mini "${gpus}" || echo "== mini fused FAILED"
mkdir -p "${WORK}/logs/mini_fused"
mv "${WORK}"/logs/tricycle_mini_*gpus_*.log "${WORK}/logs/mini_fused/" 2>/dev/null

export TRICYCLE_LEVELS=attn_out=5,ff1=5,ff2=5 TRICYCLE_PROFILE="${profile}"
for group in qkv ff1 ff2 "reshape,qk,pv,attn_out"; do
    for variant in current fused; do
        fused=""
        [[ "${variant}" == fused ]] && fused=all
        ok=0
        for vregs in 16384 24576; do
            echo "== base ${group} ${variant}: ${vregs} vregs"
            if TRICYCLE_FUSED="${fused}" TRICYCLE_MATMULS="${group}" CERIUM_VREGS="${vregs}" \
                CERIUM_BCUS="${CERIUM_BCUS:-1024}" bash "${SCRIPT_DIR}/run_pod.sh" base "${gpus}"; then
                ok=1
                break
            fi
        done
        (( ok )) || echo "== base ${group} ${variant} FAILED"
        mkdir -p "${WORK}/logs/ab/${variant}"
        mv "${WORK}"/logs/tricycle_base_*gpus* "${WORK}/logs/ab/${variant}/" 2>/dev/null
    done
done
echo "== ab done"
