#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# BERT-Base split programs compiled the way Cerium's BERT-Base compiles its
# weight matmuls (CERIUM_BLOCK_SIZE=32, 128 base-conversion units), current
# versus fused plaintext products, each with warm timing and an nsys
# breakdown. Run after ab_fused.sh has built Cerium. Run it under nohup.
#
# Usage: bash ab_blocksize.sh [BLOCK_SIZE] [GROUPS...]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${CERIUM_WORK:-/workspace/cerium_work}"
block="${1:-32}"
shift || true
groups=("$@")
(( ${#groups[@]} )) || groups=(qkv ff1 ff2 "reshape,qk,pv,attn_out")

export CERIUM_BLOCK_SIZE="${block}" CERIUM_BCUS="${CERIUM_BCUS:-128}"
export TRICYCLE_LEVELS=attn_out=5,ff1=5,ff2=5 TRICYCLE_PROFILE=1
out="${WORK}/logs/bs${block}"
for group in "${groups[@]}"; do
    for variant in current fused; do
        fused=""
        [[ "${variant}" == fused ]] && fused=all
        ok=0
        for vregs in 8192 16384 24576 32768; do
            echo "== base ${group} ${variant} block ${block}: ${vregs} vregs"
            if TRICYCLE_FUSED="${fused}" TRICYCLE_MATMULS="${group}" CERIUM_VREGS="${vregs}" \
                bash "${SCRIPT_DIR}/run_pod.sh" base 1; then
                ok=1
                break
            fi
        done
        (( ok )) || echo "== base ${group} ${variant} FAILED"
        mkdir -p "${out}/${variant}"
        mv "${WORK}"/logs/tricycle_base_*gpus* "${out}/${variant}/" 2>/dev/null
    done
done
echo "== ab_blocksize done"
