#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# BERT-Base on one 80 GB GPU, a few products per program. All seven in one
# program need a 32768-vreg register file (~66 GB) plus ~32 GB of weight
# diagonals, which runs out of device memory when the inputs are copied.
#
# Usage: bash run_base_split.sh [GPUS]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpus="${1:-1}"
groups=("reshape,qk,pv,attn_out" "qkv" "ff1" "ff2")
status=0
for group in "${groups[@]}"; do
    done_ok=0
    for vregs in 16384 24576; do
        echo "== base ${group}: ${vregs} vregs"
        if TRICYCLE_MATMULS="${group}" CERIUM_VREGS="${vregs}" CERIUM_BCUS="${CERIUM_BCUS:-1024}" \
            bash "${SCRIPT_DIR}/run_pod.sh" base "${gpus}"; then
            done_ok=1
            break
        fi
    done
    (( done_ok )) || { echo "== base ${group} FAILED"; status=1; }
done
echo "== base split done (status ${status})"
exit "${status}"
