#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Each nonlinear kernel as its own program: compile with increasing vregs until
# the register allocator succeeds, then run (warm timing, errors) and profile.
# Usage: bash stage2.sh "TAG|SPEC|LIMBS|LEVEL" ...
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${CERIUM_WORK:-/workspace/cerium_work}"
logs="${WORK}/logs/nonlinear"
mkdir -p "${logs}"

for job in "$@"; do
    IFS='|' read -r tag spec limbs level <<< "${job}"
    for vregs in ${VREGS_LADDER:-8192 16384 32768}; do
        echo "== ${tag}: ${spec}, limbs ${limbs}, level ${level}, vregs ${vregs} $(date)"
        if TRICYCLE_NONLINEAR="${spec}" TRICYCLE_LIMBS="${limbs}" TRICYCLE_INPUT_LEVEL="${level}" \
            CERIUM_VREGS="${vregs}" TRICYCLE_PROFILE=1 bash "${SCRIPT_DIR}/run_pod.sh" "${tag}"; then
            grep -h "warm\|worst\|NumPy program vs" "${logs}/${tag}_run.log"
            break
        fi
        grep -q "Evict Reg" "${logs}/${tag}_compile.log" || { echo "== ${tag} failed (not registers)"; tail -5 "${logs}/${tag}_compile.log"; break; }
    done
done
echo "== stage2 done $(date)"
