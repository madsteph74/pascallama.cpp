#!/usr/bin/env bash
# setup.sh — Clone upstream llama.cpp and apply pascal patches
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/llama.cpp"

if [ -d "${UPSTREAM_DIR}/.git" ]; then
    echo "✓ Upstream llama.cpp already exists, updating..."
    cd "${UPSTREAM_DIR}"
    git fetch origin
    git reset --hard origin/main
else
    echo "Cloning upstream llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "${UPSTREAM_DIR}"
fi

# Apply patches
echo "Applying pascal patches..."
cd "${UPSTREAM_DIR}"
git apply --check "${SCRIPT_DIR}/patches/*.patch" 2>&1 && \
    git apply "${SCRIPT_DIR}/patches/*.patch" || {
        echo "⚠ Some patches failed to apply. Check for conflicts."
        exit 1
    }

echo "✓ Setup complete."
echo "  Run '${UPSTREAM_DIR}/../build.sh' or './build.sh' from pascallama.cpp root to compile."
