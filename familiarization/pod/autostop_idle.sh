#!/usr/bin/env bash
# Stop this RunPod pod after IDLE_MINUTES with no python3, nvcc, cc1plus, cmake,
# make or apt process running. Run detached: nohup bash autostop_idle.sh &
eval "$(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(POD_ID|API_KEY)=" | sed "s/^/export /")"
IDLE_MINUTES="${IDLE_MINUTES:-45}"
LOG="${LOG:-/workspace/cerium_work/autostop.log}"
idle=0
while true; do
    if pgrep -x 'python3|python3.12|nvcc|cc1plus|cmake|make|apt-get|dpkg|nsys' >/dev/null; then
        idle=0
    else
        idle=$((idle + 1))
    fi
    if (( idle >= IDLE_MINUTES )); then
        echo "autostop (idle ${IDLE_MINUTES} min) at $(date)" >> "${LOG}"
        runpodctl stop pod "${RUNPOD_POD_ID}" >> "${LOG}" 2>&1
        exit 0
    fi
    sleep 60
done
