#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Installing vLLM 0.9.1 (CPU) - Editable Mode             ║"
echo "╚════════════════════════════════════════════════════════════╝"

VLLM_VERSION="0.9.1"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-./vllm_source}"

# Detect if we're in a uv environment
if command -v uv &> /dev/null; then
    PIP="uv pip"
    INDEX_STRATEGY="--index-strategy unsafe-best-match"
else
    PIP="python -m pip"
    INDEX_STRATEGY=""
fi

# Clone vLLM if not already present
if [ -d "$VLLM_SOURCE_DIR" ]; then
    echo "→ vLLM source directory already exists: $VLLM_SOURCE_DIR"
    read -p "Remove and re-clone? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VLLM_SOURCE_DIR"
    else
        echo "→ Using existing source directory"
    fi
fi

if [ ! -d "$VLLM_SOURCE_DIR" ]; then
    echo "→ Cloning vLLM v${VLLM_VERSION}..."
    git clone --depth 1 --branch "v${VLLM_VERSION}" https://github.com/vllm-project/vllm.git "$VLLM_SOURCE_DIR"
fi

cd "$VLLM_SOURCE_DIR"

# Install Python build dependencies (only if not already installed)
echo "→ Installing Python build dependencies..."
$PIP install --upgrade pip wheel packaging ninja "setuptools>=49.4.0" "setuptools_scm>=8" cmake

# Install vLLM CPU requirements
echo "→ Installing vLLM ${VLLM_VERSION} requirements..."

# Check current transformers version before installation
if python -c "import transformers; print(transformers.__version__)" 2>/dev/null; then
    BEFORE_VERSION=$(python -c "import transformers; print(transformers.__version__)")
    echo "  → Current transformers: ${BEFORE_VERSION}"
fi

# Install all requirements
# Note: Some packages may be reinstalled, but this ensures compatibility
if [ -f "requirements-cpu.txt" ]; then
    $PIP install -v -r requirements-cpu.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        $INDEX_STRATEGY
elif [ -f "requirements/cpu.txt" ]; then
    $PIP install -v -r requirements/cpu.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        $INDEX_STRATEGY
else
    echo "❌ Could not find requirements file"
    exit 1
fi

# Check if transformers changed
if python -c "import transformers; print(transformers.__version__)" 2>/dev/null; then
    AFTER_VERSION=$(python -c "import transformers; print(transformers.__version__)")
    if [ "$BEFORE_VERSION" != "$AFTER_VERSION" ] && [ -n "$BEFORE_VERSION" ]; then
        echo "  ⚠️  transformers changed: ${BEFORE_VERSION} → ${AFTER_VERSION}"
        echo "  To keep your pinned version, reinstall after vLLM build:"
        echo "    uv pip install 'transformers>=4.43,<4.54.0'"
    fi
fi

# Build and install vLLM in editable mode
echo "→ Building and installing vLLM in editable mode (this may take 10-15 minutes)..."
export VLLM_TARGET_DEVICE=cpu
export MAX_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export CCACHE_DIR="${HOME}/.cache/ccache"

$PIP install -e . --no-build-isolation

cd ..

# Re-install user's pinned transformers if vLLM changed it
if [ -n "$BEFORE_VERSION" ] && [ -n "$AFTER_VERSION" ] && [ "$BEFORE_VERSION" != "$AFTER_VERSION" ]; then
    echo ""
    echo "→ Restoring your pinned transformers version..."
    $PIP install 'transformers>=4.43,<4.54.0' --upgrade
    FINAL_VERSION=$(python -c "import transformers; print(transformers.__version__)")
    echo "  ✓ Restored transformers ${FINAL_VERSION}"
fi

# ✅ Ensure torchvision is not installed
echo "→ Removing torchvision (if installed)..."
pip3 uninstall -y torchvision || true

# Verify installation
echo "→ Verifying vLLM installation..."
python -c "import vllm; print(f'✅ vLLM {vllm.__version__} installed successfully (editable mode)')"
python -c "import transformers; print(f'   Using transformers {transformers.__version__}')"
python -c "import vllm; import os; print(f'Source location: {os.path.dirname(vllm.__file__)}')"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ vLLM ${VLLM_VERSION} (CPU) installed in editable mode! ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Source directory: $(realpath $VLLM_SOURCE_DIR)"
echo "You can now modify vLLM source and changes will take effect immediately."