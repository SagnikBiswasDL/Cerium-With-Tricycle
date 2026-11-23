#!/usr/bin/env bash
# Cerium's own BERT-Base benchmark (benchmarks/bert_base) on one GPU of a
# Runpod Pod set up with setup_pod.sh: the baseline for the Tricycle work.
#
# Same steps as the benchmark's compile.sh + run_bert_benchmarks.sh, without
# Docker and for one GPU only (make_bert.sh builds all of 1/2/4/8 and stops at
# the first count that was not compiled). The weights are packed with the
# PyTorch that ships in the runpod/pytorch image, so the venv stays minimal.
#
# Usage: bash run_bert_baseline.sh [NUM_SAMPLES]     default 5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${CERIUM_WORK:-/workspace/cerium_work}"
samples="${1:-5}"

# Some data centers time out on plain-HTTP archive.ubuntu.com.
sed -i 's#http://archive.ubuntu.com/ubuntu/#https://archive.ubuntu.com/ubuntu/#' \
    /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
JOBS="${JOBS:-$(nproc)}" bash "${SCRIPT_DIR}/setup_pod.sh" "${CUDA_ARCH:-90}"
source "${WORK}/env.sh"

python="${CERIUM_REPO}/venv_cerium/bin/python3"
bench="${CERIUM_REPO}/benchmarks/bert_base"
logs="${WORK}/logs"
mkdir -p "${logs}"
cd "${bench}"

model="bert_torch_rte.pt"
model_sha256="eeefe342a8476810c8a13014abfcaa115909bfd2cc7082015b16fc19400539c9"
if [[ ! -f "${model}" ]]; then
    curl --fail --location --silent --show-error --output "${model}" \
        "https://drive.usercontent.google.com/download?id=1dOFhA1YR_hiqVEFP_Q4DVo3Dx6oESAKZ&export=download&authuser=0&confirm=t"
fi
echo "${model_sha256}  ${model}" | sha256sum --check -
if [[ ! -f packed_params/params_rte_block12.pkl ]]; then
    echo "== Packing weights"
    /usr/bin/python3 bert_params_torch.py --output-dir packed_params --task-name rte
fi
"${python}" -c "import tqdm" 2>/dev/null || "${python}" -m pip install -q tqdm

start=$(date +%s)
if [[ ! -f bert_outputs/main_1gpu_12/program_inputs ]]; then
    echo "== Compiling (Cerium compiler)"
    "${python}" bert_cerium.py --gpus 1 2>&1 | tee "${logs}/bert_compile_1gpu.log" | grep -a "^Compiling"
fi
# main_1gpu_12 and everything it calls (bert_outputs/*/calls); the *_vec_2 and
# *_vec2 variants are generated only for multi-GPU builds.
functions=(bootstrap_32K_33lvl_t1 bootstrap_32K_35lvl_t1_vec_1
    matmul_ct128x128_ct128x128 matmul_ct128x128_ct128x128_transpose gelu_vec
    pool_classify_layer softmax_128x128_vec layernorm_att_1gpu layernorm_ffn_1gpu
    attention_qkv_1gpu attention_o_1gpu ffn_up_1gpu ffn_down_1gpu main_1gpu_12)
echo "== Building CUDA for ${#functions[@]} functions"
for function_name in "${functions[@]}"; do
    make all -j"${MAKE_JOBS:-16}" FNAME="${function_name}" DIRNAME=bert_outputs >> "${logs}/bert_compile_1gpu.log" 2>&1
    mkdir -p "bert_outputs/compiled/${function_name}"
    cp -r "bert_outputs/${function_name}/output/." "bert_outputs/compiled/${function_name}/"
done
echo "== Compiled in $(( $(date +%s) - start ))s"

echo "== Running ${samples} samples on 1 GPU"
"${python}" bert_runner.py --gpus 1 --num-samples "${samples}" 2>&1 | tee "${logs}/bert_run_1gpu.log" \
    | grep -aE "Bert Execution time|Prediction|Actual|Final Accuracy|Error|error"
awk '/Bert Execution time:/ { sum += $4; n++ } END { if (n) printf "== Average execution time over %d samples: %.3f s\n", n, sum / n }' \
    "${logs}/bert_run_1gpu.log"
