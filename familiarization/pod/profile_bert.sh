#!/usr/bin/env bash
# GPU time per Cerium function for one sample of Cerium's BERT-Base benchmark,
# after run_bert_baseline.sh has compiled it. Cerium names each kernel
# Cerium::<function>::..., so summing Nsight Systems' per-kernel totals by that
# namespace splits the 12-layer run into QKV, attention output, feed-forward,
# softmax, bootstrapping and so on.
#
# Usage: bash profile_bert.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${CERIUM_WORK:-/workspace/cerium_work}"
sed -i 's#http://archive.ubuntu.com/ubuntu/#https://archive.ubuntu.com/ubuntu/#' \
    /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
JOBS="${JOBS:-$(nproc)}" bash "${SCRIPT_DIR}/setup_pod.sh" "${CUDA_ARCH:-90}"
source "${WORK}/env.sh"

if ! command -v nsys >/dev/null; then
    apt-get install -y -qq cuda-nsight-systems-13-3 >/dev/null 2>&1 \
        || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nsight-systems >/dev/null
fi
nsys="$(command -v nsys || ls /usr/local/cuda-13.3/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)"
echo "== nsys: ${nsys}"

python="${CERIUM_REPO}/venv_cerium/bin/python3"
out="${WORK}/profile"
mkdir -p "${out}"
cd "${CERIUM_REPO}/benchmarks/bert_base"
"${nsys}" profile --trace=cuda --cuda-graph-trace=node --force-overwrite=true \
    --output="${out}/bert_1sample" \
    "${python}" bert_runner.py --gpus 1 --num-samples 1 2>&1 \
    | grep -aE "Bert Execution time|Prediction|Actual|Final Accuracy|Error|error"
"${nsys}" stats --report cuda_gpu_kern_sum --format csv --force-export=true \
    --output="${out}/bert_1sample" "${out}/bert_1sample.nsys-rep" > /dev/null

"${python}" - "${out}/bert_1sample_cuda_gpu_kern_sum.csv" <<'EOF'
import csv, re, sys
from collections import defaultdict
total_ns = defaultdict(float)
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        name = row["Name"]
        match = re.search(r"Cerium::(\w+?)::", name)
        total_ns[match.group(1) if match else "runtime (unnamed)"] += float(row["Total Time (ns)"])
grand = sum(total_ns.values())
print(f"{'function':45s} {'GPU ms':>10s} {'share':>7s}")
for name, ns in sorted(total_ns.items(), key=lambda kv: -kv[1]):
    print(f"{name:45s} {ns / 1e6:10.1f} {100 * ns / grand:6.1f}%")
print(f"{'total':45s} {grand / 1e6:10.1f}")
EOF
