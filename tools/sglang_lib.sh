# Shared SGLang version / base-commit / patch-path resolution for TorchSpec tooling.
#
# After sourcing, the following variables are available:
#   SGLANG_VERSION       - e.g. v0.5.14
#   SGLANG_DIR           - docker/sglang/<version> dir holding the pin metadata
#   SGLANG_COMMIT        - base commit the patch is generated against (may be empty)
#   SGLANG_FOLDER_NAME   - local sglang checkout folder name (default _sglang)
#   SGLANG_PATH          - absolute path to the local sglang checkout
#   SGLANG_PATCH_DIR     - patches/sglang/<version>
#   SGLANG_PATCH_FILE    - patches/sglang/<version>/sglang.patch

if [ -z "${PROJECT_ROOT:-}" ]; then
    echo "Error: PROJECT_ROOT must be set before sourcing sglang_lib.sh" >&2
    return 1 2>/dev/null || exit 1
fi

: "${SGLANG_VERSION:=v0.5.14}"
: "${SGLANG_FOLDER_NAME:=_sglang}"

SGLANG_DIR="$PROJECT_ROOT/docker/sglang/$SGLANG_VERSION"

sglang_base_commit() {
    if [ -f "$SGLANG_DIR/SGLANG_COMMIT" ]; then
        tr -d '[:space:]' < "$SGLANG_DIR/SGLANG_COMMIT"
    fi
}
SGLANG_COMMIT="$(sglang_base_commit)"

# Absolute path to the local sglang working tree (built by build_conda.sh).
case "$SGLANG_FOLDER_NAME" in
    /*) SGLANG_PATH="$SGLANG_FOLDER_NAME" ;;
    *)  SGLANG_PATH="$PROJECT_ROOT/$SGLANG_FOLDER_NAME" ;;
esac

SGLANG_PATCH_DIR="$PROJECT_ROOT/patches/sglang/$SGLANG_VERSION"
SGLANG_PATCH_FILE="$SGLANG_PATCH_DIR/sglang.patch"

echo "SGLANG_VERSION: $SGLANG_VERSION"
echo "SGLANG_DIR: $SGLANG_DIR"
echo "SGLANG_COMMIT: $SGLANG_COMMIT"
echo "SGLANG_FOLDER_NAME: $SGLANG_FOLDER_NAME"
echo "SGLANG_PATH: $SGLANG_PATH"
echo "SGLANG_PATCH_DIR: $SGLANG_PATCH_DIR"
echo "SGLANG_PATCH_FILE: $SGLANG_PATCH_FILE"
