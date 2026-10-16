#!/usr/bin/env bash
# build.sh — Build pascallama with Pascal-optimised flags (validated against the Oct 2026 upstream)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/llama.cpp"

if [ ! -d "${UPSTREAM_DIR}" ]; then
    echo "Error: upstream llama.cpp not found. Run './setup.sh' first."
    exit 1
fi

cd "${UPSTREAM_DIR}"

echo "Building pascallama.cpp (CUDA, MMQ forced, SM 6.1)..."
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release --parallel "$(nproc)"

echo "✓ Build complete."
echo "  Server: ./llama.cpp/build/bin/llama-server"
echo "  Speed check: ./llama.cpp/build/bin/llama-cli -m <model.gguf> -ngl 99 -p 'test' -n 32"