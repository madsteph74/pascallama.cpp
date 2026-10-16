#!/usr/bin/env bash
# setup.sh — Clone upstream llama.cpp at the TESTED commit and apply the Pascal MMQ patches
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/llama.cpp"
PINNED_COMMIT="a0e6dc7384edb8f5d927155e1d26c10b4a310e6d"
UPSTREAM_URL="https://github.com/ggml-org/llama.cpp"

if [ -d "${UPSTREAM_DIR}/.git" ]; then
    echo "✓ Upstream llama.cpp already exists, fetching and checking out the pinned commit..."
    cd "${UPSTREAM_DIR}"
    git fetch origin || true
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        echo "⚠ WARNING: your upstream directory has local modifications."
        echo "  The patches are probably already applied. To start over:"
        echo "    cd ${UPSTREAM_DIR} && git checkout -- . && git checkout ${PINNED_COMMIT}"
        echo "  Press Enter to continue anyway, or Ctrl+C to abort."
        read -r
    fi
else
    echo "Cloning upstream llama.cpp (full history is needed for the pinned commit)..."
    git clone "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
fi

cd "${UPSTREAM_DIR}"
echo "Checking out the tested commit ${PINNED_COMMIT}..."
git checkout --quiet "${PINNED_COMMIT}"

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "Working tree dirty before applying; undoing tracked changes..."
    git checkout -- .
fi

echo "Applying the Pascal MMQ patches..."
git apply --check "${SCRIPT_DIR}/patches/001-pascal-mmq-dp4a.patch" || {
    echo "⚠ The patch no longer applies cleanly at the pinned commit."
    echo "  If upstream moved the files, the hunks need updating (see README.md)."
    echo "  Patch file: ${SCRIPT_DIR}/patches/001-pascal-mmq-dp4a.patch"
    exit 1
}
git apply "${SCRIPT_DIR}/patches/001-pascal-mmq-dp4a.patch"

echo "✓ Setup complete (llama.cpp @ ${PINNED_COMMIT} + Pascal MMQ patches)."
echo "  To undo the patches:  git apply -R ../patches/001-pascal-mmq-dp4a.patch"
echo "  To re-apply cleanly:  git checkout -- ."
echo "  Run './build.sh' to compile."