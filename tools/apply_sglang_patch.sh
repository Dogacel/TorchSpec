#!/bin/bash

# Apply the TorchSpec sglang patch onto a local sglang checkout.
#
# Usage:
#   ./tools/apply_sglang_patch.sh <path-to-sglang-repo>
#
# Please note that this will overwrite all local changes and delete untracked files.

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=tools/sglang_lib.sh
source "$SCRIPT_DIR/sglang_lib.sh"


if [ ! -d "$SGLANG_PATH" ]; then
    echo "Error: $SGLANG_PATH directory not found"
    exit 1
fi

cd "$SGLANG_PATH"

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: $SGLANG_PATH is not a git repository"
    exit 1
fi

if ! git rev-parse "$SGLANG_COMMIT" > /dev/null 2>&1; then
    echo "Error: Commit $SGLANG_COMMIT not found in $SGLANG_PATH repository"
    exit 1
fi

echo "Resetting to base commit $SGLANG_COMMIT..."
git reset --hard "$SGLANG_COMMIT"
git clean -fd

echo ""
echo "Applying patch..."
git apply "$SGLANG_PATCH_FILE"

echo ""
echo "✓ Patch applied successfully."
echo ""
echo "Files modified:"
git status --short
