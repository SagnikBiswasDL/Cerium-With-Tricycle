#!/usr/bin/env bash
# Native (no Docker) equivalent of Cerium's docker/Dockerfile + build_cerium.sh
# + activate_cerium_venv.sh for a RunPod pod (Ubuntu 24.04, x86_64, NVIDIA GPU).
#
# Everything except the apt packages lives under $WORK on the persistent
# /workspace volume. RunPod wipes the container disk when a pod is stopped, so
# rerun this script after a restart; finished steps are skipped.
#
# Usage: bash setup_pod.sh [CUDA_ARCH]     (default 90 = Hopper H100/H200)
set -euo pipefail

CUDA_ARCH="${1:-90}"
WORK="${CERIUM_WORK:-/workspace/cerium_work}"
DEPS="${WORK}/deps"
REPO="${WORK}/Cerium"
CUDA_HOME=/usr/local/cuda-13.3
NCCL_VERSION=2.30.7-1+cuda13.3
SEAL_VERSION=v4.1.1
JOBS="${JOBS:-64}"

mkdir -p "${DEPS}"
log() { echo "[setup_pod] $*"; }

# 1. System packages (Dockerfile steps 2 and 5). The CUDA 13.3 toolkit installs
# beside any existing CUDA; keep the pod's default /usr/local/cuda unchanged.
previous_cuda="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]] || ! dpkg -s g++-12 python3.12-dev python3.12-venv >/dev/null 2>&1; then
    log "Installing CUDA 13.3 toolkit, g++-12, Python 3.12 headers"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cuda-toolkit-13-3 g++-12 python3.12-dev python3.12-venv m4 zlib1g-dev git cmake make
fi
if [[ -n "${previous_cuda}" && "$(readlink -f /usr/local/cuda)" != "${previous_cuda}" ]]; then
    log "Restoring /usr/local/cuda -> ${previous_cuda}"
    update-alternatives --set cuda "${previous_cuda}" || true
fi

# 2. NCCL 2.30.7 for CUDA 13.3, extracted locally instead of replacing the
# system NCCL (Dockerfile step 7).
NCCL_DIR="${DEPS}/nccl-${NCCL_VERSION}"
if [[ ! -f "${NCCL_DIR}/usr/include/nccl.h" ]]; then
    log "Fetching NCCL ${NCCL_VERSION}"
    tmp="$(mktemp -d)"
    (cd "${tmp}" && apt-get download "libnccl2=${NCCL_VERSION}" "libnccl-dev=${NCCL_VERSION}")
    mkdir -p "${NCCL_DIR}"
    for deb in "${tmp}"/*.deb; do dpkg -x "${deb}" "${NCCL_DIR}"; done
    rm -rf "${tmp}"
fi
NCCL_LIB_DIR="${NCCL_DIR}/usr/lib/x86_64-linux-gnu"

# 3. Microsoft SEAL 4.1.1 (docker/scripts/install_seal.sh), installed to a prefix.
SEAL_PREFIX="${DEPS}/seal-${SEAL_VERSION}"
if [[ ! -d "${SEAL_PREFIX}/lib/cmake" ]]; then
    log "Building SEAL ${SEAL_VERSION}"
    rm -rf "${DEPS}/SEAL"
    git clone --quiet https://github.com/microsoft/SEAL.git "${DEPS}/SEAL"
    git -C "${DEPS}/SEAL" checkout --quiet "${SEAL_VERSION}"
    cmake -S "${DEPS}/SEAL" -B "${DEPS}/SEAL/build" -DCMAKE_INSTALL_PREFIX="${SEAL_PREFIX}"
    cmake --build "${DEPS}/SEAL/build" -j "${JOBS}"
    cmake --install "${DEPS}/SEAL/build"
fi

# 4. Cerium source with submodules.
if [[ ! -d "${REPO}/.git" ]]; then
    log "Cloning Cerium"
    git clone --recurse-submodules https://github.com/CMU-CAOS/Cerium.git "${REPO}"
fi
git -C "${REPO}" submodule update --init --recursive

# 5. Compiler + runtime (build_cerium.sh's Python 3.12 pass; the venv uses 3.12).
if [[ ! -f "${REPO}/build/cerium/runtime/libCeriumRuntime.so" ]]; then
    log "Building Cerium for CUDA arch ${CUDA_ARCH}"
    cmake -S "${REPO}" -B "${REPO}/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython3_EXECUTABLE=/usr/bin/python3.12 \
        -DCERIUM_DEBUG_EXEC=OFF \
        -DCERIUM_DEBUG_COMPILER=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
        -DCUDAToolkit_ROOT="${CUDA_HOME}" \
        -DCMAKE_PREFIX_PATH="${SEAL_PREFIX}" \
        -DNCCL_INCLUDE_DIR="${NCCL_DIR}/usr/include" \
        -DNCCL_LIBRARY="${NCCL_LIB_DIR}/libnccl.so"
    cmake --build "${REPO}/build" --target all -j "${JOBS}"
fi

# 6. Python environment (activate_cerium_venv.sh), with retries for flaky networks.
if [[ ! -x "${REPO}/venv_cerium/bin/python3" ]]; then
    python3.12 -m venv "${REPO}/venv_cerium"
fi
if ! (cd "${REPO}" && venv_cerium/bin/python3 -c "import numpy, cerium" 2>/dev/null); then
    log "Installing Python requirements"
    (cd "${REPO}" && venv_cerium/bin/python3 -m pip install --retries 10 --timeout 120 -r requirements.txt)
fi

# 7. Environment used by run_pod.sh.
cat > "${WORK}/env.sh" <<EOF
export CERIUM_REPO="${REPO}"
export PYTHONPATH="${REPO}:${REPO}/build/cerium/python"
export LD_LIBRARY_PATH="${REPO}/build/cerium/runtime:${NCCL_LIB_DIR}:${CUDA_HOME}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export CUDA_ARCHITECTURES="${CUDA_ARCH}"
EOF
log "Done. Environment written to ${WORK}/env.sh"
