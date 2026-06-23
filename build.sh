#!/usr/bin/env bash
# build.sh — Build pascallama with Pascal-optimised flags
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/llama.cpp"

if [ ! -d "${UPSTREAM_DIR}" ]; then
    echo "Error: upstream llama.cpp not found. Run './setup.sh' first."
    exit 1
fi

cd "${UPSTREAM_DIR}"

echo "Building pascallama.cpp..."
cmake -B build \
    -DGGML_CUDA=ON \
    -DFORCE_MMQ=ON \
    -DFA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release --parallel $(nproc)

echo "✓ Build complete."
echo "  Binary: ${UPSTREAM_DIR}/build/bin/llama-server"
