#!/bin/bash

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Parse command line arguments
# Usage: ./build_conda.sh [MODE] [BACKEND]
#   MODE:
#     1       - Create a new micromamba/conda env and install (default)
#     current - Install into current environment
#     0       - Skip env creation and installation
#   BACKEND:
#     sglang     - Install SGLang only (default)
#     vllm       - Install vLLM only
#     tokenspeed - Install TokenSpeed from an editable source checkout
#     both       - Install both backends

MODE="${1:-1}"
BACKEND="${2:-sglang}"

# Validate backend
if [[ ! "$BACKEND" =~ ^(sglang|vllm|tokenspeed|both)$ ]]; then
    echo "Error: Invalid backend '$BACKEND'"
    echo "Usage: $0 [MODE] [BACKEND]"
    echo "  BACKEND options: sglang (default), vllm, tokenspeed, both"
    exit 1
fi

echo "=========================================="
echo "TorchSpec Installation"
echo "Backend: $BACKEND"
echo "=========================================="

ENV_MANAGER=""
ENV_CREATE_CMD=()
ENV_RUN_CMD=()
ACTIVATE_HINT=""

if command -v micromamba &> /dev/null; then
    ENV_MANAGER="micromamba"
    export MAMBA_EXE="${MAMBA_EXE:-$(command -v micromamba)}"
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    ENV_CREATE_CMD=("$MAMBA_EXE" create -n torchspec python=3.12 pip uv -c conda-forge -y)
    ENV_RUN_CMD=("$MAMBA_EXE" run -n torchspec)
    ACTIVATE_HINT="micromamba activate torchspec"
elif command -v conda &> /dev/null; then
    ENV_MANAGER="conda"
    ENV_CREATE_CMD=(conda create -n torchspec python=3.12 pip uv -c conda-forge -y)
    ENV_RUN_CMD=(conda run -n torchspec)
    ACTIVATE_HINT="conda activate torchspec"
fi

if [ "$MODE" = "1" ]; then
    if [ -z "$ENV_MANAGER" ]; then
        echo "Error: neither micromamba nor conda is installed."
        echo "Please install one of them first:"
        echo "  micromamba: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
        echo "  conda:      https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html"
        exit 1
    fi

    "${ENV_CREATE_CMD[@]}"
elif [ "$MODE" = "current" ]; then
    echo "Using current environment: $(python --version), $(command -v python)"
else
    echo "Skipping environment setup (mode=0)"
fi

# Install SGLang if requested
if [ "$BACKEND" = "sglang" ] || [ "$BACKEND" = "both" ]; then
    echo "=========================================="
    echo "Installing SGLang..."
    echo "=========================================="

    # Resolve SGLANG_VERSION / SGLANG_COMMIT / paths from the single source of truth.
    # shellcheck source=tools/sglang_lib.sh
    source "$SCRIPT_DIR/sglang_lib.sh"

    if [ -z "$SGLANG_COMMIT" ]; then
        echo "Error: Could not find base commit in $SGLANG_DIR/SGLANG_COMMIT"
        exit 1
    fi

    # Install sglang inside the conda environment
    if [ ! -d "$SGLANG_PATH" ]; then
        git clone https://github.com/sgl-project/sglang.git "$SGLANG_PATH"
    fi

    # Avoid pythonpath conflict, because we are using the offline engine.
    cd "$SGLANG_PATH"
    git checkout $SGLANG_COMMIT
    git reset --hard HEAD

    cd "$PROJECT_ROOT"

    if [ "$MODE" = "1" ]; then
        "${ENV_RUN_CMD[@]}" pip install -e "${SGLANG_FOLDER_NAME}/python[all]"
    elif [ "$MODE" = "current" ]; then
        pip install -e "${SGLANG_FOLDER_NAME}/python[all]"
    fi

    cd "$SGLANG_PATH"

    # Apply sglang patch (matches Docker build behavior)
    git apply "$SGLANG_PATCH_FILE"

    cd "$PROJECT_ROOT"
fi

# Install vLLM if requested
if [ "$BACKEND" = "vllm" ] || [ "$BACKEND" = "both" ]; then
    echo "=========================================="
    echo "Installing vLLM..."
    echo "=========================================="

    if [ "$MODE" = "1" ]; then
        "${ENV_RUN_CMD[@]}" uv pip install "vllm>=0.16.0"
    elif [ "$MODE" = "current" ]; then
        pip install "vllm>=0.16.0"
    fi
fi

# Install TokenSpeed if requested
if [ "$BACKEND" = "tokenspeed" ] && [ "$MODE" != "0" ]; then
    echo "=========================================="
    echo "Installing TokenSpeed..."
    echo "=========================================="

    TOKENSPEED_REPO="${TOKENSPEED_REPO:-https://github.com/lightseekorg/tokenspeed.git}"
    TOKENSPEED_FOLDER_NAME="${TOKENSPEED_FOLDER_NAME:-_tokenspeed}"
    TOKENSPEED_PATH="${TOKENSPEED_PATH:-$PROJECT_ROOT/$TOKENSPEED_FOLDER_NAME}"
    TOKENSPEED_REF="${TOKENSPEED_REF:-}"

    if [[ "$TOKENSPEED_PATH" != /* ]]; then
        TOKENSPEED_PATH="$PROJECT_ROOT/$TOKENSPEED_PATH"
    fi

    if [ -e "$TOKENSPEED_PATH" ] && [ ! -d "$TOKENSPEED_PATH/.git" ]; then
        echo "Error: TOKENSPEED_PATH exists but is not a git checkout: $TOKENSPEED_PATH"
        exit 1
    fi

    if [ ! -d "$TOKENSPEED_PATH/.git" ]; then
        git clone "$TOKENSPEED_REPO" "$TOKENSPEED_PATH"
    else
        echo "Reusing existing TokenSpeed checkout: $TOKENSPEED_PATH"
    fi

    if [ -n "$TOKENSPEED_REF" ]; then
        echo "Checking out requested TokenSpeed ref: $TOKENSPEED_REF"
        git -C "$TOKENSPEED_PATH" checkout "$TOKENSPEED_REF"
    fi

    # TokenSpeed's native packages currently target Python 3.12. In particular,
    # the kernel and CUDA dependency wheels do not resolve on Python 3.14.
    TOKENSPEED_PYTHON_CHECK="import sys; assert sys.version_info[:2] == (3, 12), \
f'TokenSpeed requires Python 3.12, got {sys.version.split()[0]}'"

    # Match TokenSpeed's published development-install instructions. The
    # variable is needed by the runner image's system Python and is harmless in
    # an isolated conda environment.
    export PIP_BREAK_SYSTEM_PACKAGES=1

    # Follow TokenSpeed's NVIDIA Docker build order. Installing the in-tree
    # kernel first satisfies the runtime's tokenspeed-kernel>=0.1.3.dev0
    # dependency without trying to resolve an unavailable development wheel.
    if [ "$MODE" = "1" ]; then
        "${ENV_RUN_CMD[@]}" python -c "$TOKENSPEED_PYTHON_CHECK"
        "${ENV_RUN_CMD[@]}" python -m pip install "setuptools==69.5.1" wheel
        "${ENV_RUN_CMD[@]}" python -m pip install \
            -e "$TOKENSPEED_PATH/tokenspeed-kernel/python" \
            --no-build-isolation
        "${ENV_RUN_CMD[@]}" python -m pip install \
            -e "$TOKENSPEED_PATH/tokenspeed-scheduler"
        "${ENV_RUN_CMD[@]}" python -m pip install \
            -e "$TOKENSPEED_PATH/python" \
            --no-build-isolation
    elif [ "$MODE" = "current" ]; then
        python -c "$TOKENSPEED_PYTHON_CHECK"
        python -m pip install "setuptools==69.5.1" wheel
        python -m pip install \
            -e "$TOKENSPEED_PATH/tokenspeed-kernel/python" \
            --no-build-isolation
        python -m pip install \
            -e "$TOKENSPEED_PATH/tokenspeed-scheduler"
        python -m pip install \
            -e "$TOKENSPEED_PATH/python" \
            --no-build-isolation
    fi
fi

# Install torchspec with appropriate extras
if [ "$MODE" = "1" ]; then
    echo "=========================================="
    echo "Installing TorchSpec..."
    echo "=========================================="

    EXTRAS="dev"
    if [ "$BACKEND" = "vllm" ]; then
        EXTRAS="dev,vllm"
    elif [ "$BACKEND" = "both" ]; then
        EXTRAS="dev,vllm"
    fi

    "${ENV_RUN_CMD[@]}" uv pip install -e ".[$EXTRAS]"

    echo ""
    echo "=========================================="
    echo "✓ TorchSpec environment setup complete!"
    echo "=========================================="
    echo "Activate with: $ACTIVATE_HINT"
    echo ""
    if [ "$BACKEND" = "sglang" ]; then
        echo "Backend: SGLang"
        echo "Run: ./examples/qwen3-8b-single-node/run.sh"
    elif [ "$BACKEND" = "vllm" ]; then
        echo "Backend: vLLM"
        echo "Run: ./examples/qwen3-8b-single-node/run.sh --config configs/vllm_qwen3_8b.yaml"
    elif [ "$BACKEND" = "both" ]; then
        echo "Backends: SGLang + vLLM"
        echo "SGLang: ./examples/qwen3-8b-single-node/run.sh"
        echo "vLLM:   ./examples/qwen3-8b-single-node/run.sh --config configs/vllm_qwen3_8b.yaml"
    elif [ "$BACKEND" = "tokenspeed" ]; then
        echo "Backend: TokenSpeed"
        echo "Source: $TOKENSPEED_PATH"
    fi
elif [ "$MODE" = "current" ]; then
    EXTRAS="dev"
    if [ "$BACKEND" = "vllm" ]; then
        EXTRAS="dev,vllm"
    elif [ "$BACKEND" = "both" ]; then
        EXTRAS="dev,vllm"
    fi

    pip install -e ".[$EXTRAS]"

    echo ""
    echo "=========================================="
    echo "✓ TorchSpec installed into current environment!"
    echo "=========================================="
else
    echo ""
    echo "Skipping package installation (mode=0)"
    echo "Please install packages manually:"
    if [ "$BACKEND" = "sglang" ]; then
        echo "  pip install -e \"${SGLANG_FOLDER_NAME}/python[all]\""
        echo "  pip install -e \".[dev]\""
    elif [ "$BACKEND" = "vllm" ]; then
        echo "  pip install vllm>=0.16.0"
        echo "  pip install -e \".[dev,vllm]\""
    elif [ "$BACKEND" = "both" ]; then
        echo "  pip install -e \"${SGLANG_FOLDER_NAME}/python[all]\""
        echo "  pip install vllm>=0.16.0"
        echo "  pip install -e \".[dev,vllm]\""
    elif [ "$BACKEND" = "tokenspeed" ]; then
        echo "  git clone https://github.com/lightseekorg/tokenspeed.git _tokenspeed"
        echo "  export PIP_BREAK_SYSTEM_PACKAGES=1"
        echo "  pip install setuptools==69.5.1 wheel"
        echo "  pip install -e \"_tokenspeed/tokenspeed-kernel/python\" --no-build-isolation"
        echo "  pip install -e \"_tokenspeed/tokenspeed-scheduler\""
        echo "  pip install -e \"_tokenspeed/python\" --no-build-isolation"
        echo "  pip install -e \".[dev]\""
    fi
fi
