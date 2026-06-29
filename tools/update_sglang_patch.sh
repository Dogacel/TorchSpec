#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=tools/sglang_lib.sh
source "$SCRIPT_DIR/sglang_lib.sh"


echo "SGLANG_VERSION: $SGLANG_VERSION"
echo "SGLANG_COMMIT: $SGLANG_COMMIT"
echo "Using folder: $SGLANG_PATH"

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

# Check for uncommitted or untracked changes
has_uncommitted=false
if ! git diff --quiet HEAD 2>/dev/null; then
    has_uncommitted=true
fi
if [ -n "$(git ls-files --others --exclude-standard)" ]; then
    has_uncommitted=true
fi

if [ "$(git rev-parse HEAD)" = "$(git rev-parse $SGLANG_COMMIT)" ]; then
    echo "Error: No commits after $SGLANG_COMMIT."
    if [ "$has_uncommitted" = true ]; then
        echo ""
        echo "You have uncommitted changes:"
        git status --short
        echo ""
        echo "Please commit them first:"
        echo "  cd $SGLANG_PATH && git add -A && git commit -m 'your message'"
    else
        echo "Please make and commit your changes in $SGLANG_PATH first."
    fi
    exit 1
fi

if [ "$has_uncommitted" = true ]; then
    echo "Error: You have uncommitted changes that will NOT be included in the patch:"
    git status --short
    echo ""
    echo "Please commit them first:"
    echo "  cd $SGLANG_PATH && git add -A && git commit --amend --no-edit"
    exit 1
fi

mkdir -p "$SGLANG_PATCH_DIR"

echo "Generating patch from $SGLANG_COMMIT to HEAD..."
# Write diffstat header as a comment, then the actual diff.
# git apply ignores lines before the first "diff --git" line,
# so the diffstat is purely informational for human readers.
{
    echo "torchspec sglang patch (base: ${SGLANG_COMMIT:0:10})"
    echo "---"
    git diff --stat "$SGLANG_COMMIT" HEAD
    echo ""
    git diff "$SGLANG_COMMIT" HEAD
} > "$SGLANG_PATCH_FILE"

if [ ! -s "$SGLANG_PATCH_FILE" ]; then
    echo "Error: Failed to generate patch or patch is empty"
    exit 1
fi

PATCH_SIZE=$(wc -l < "$SGLANG_PATCH_FILE")
echo "✓ Patch updated successfully: patches/sglang/$SGLANG_VERSION/sglang.patch ($PATCH_SIZE lines)"

echo ""
echo "Files modified:"
git diff --name-status "$SGLANG_COMMIT" HEAD
