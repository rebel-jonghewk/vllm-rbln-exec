# vllm-rbln-exec

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.9.1-green.svg)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**CPU vs RBLN Parity Runner** - A tool for comparing vLLM inference outputs between CPU and RBLN (Rebellions) accelerators with logits/logprobs inspection.

## Quick Start

```bash
# 1. Install system dependencies (Ubuntu/Debian)
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 cmake ninja-build git libopenblas-dev

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: pip install uv

# 3. Clone the repository
git clone git@github.com:rebel-jonghewk/vllm-rbln-exec.git
cd vllm-rbln-exec

# 4. Create virtual environment
uv venv
source .venv/bin/activate

# 5. Install vLLM and dependencies
make install  # Takes 10-15 minutes (builds vLLM from source)

# 6. Run your first comparison
vllm-rbln-exec --model llama3.2-1b

# 7. Try with custom prompts
vllm-rbln-exec --model llama3.2-1b --prompts "Hello, world!" "Once upon a time"

# 8. Full vocabulary inspection
vllm-rbln-exec --model llama3.2-1b --logprobs -1 --max-tokens 128

# 9. MoE
vllm-rbln-exec --model qwen1.5-moe-15b --num-hidden-layers 1 --use-cache --ep --tp 4 --max-model-len 8192 --block 4096
```

**That's it!** See [Usage](#usage) for more examples.

### Alternative: Without uv

If you prefer traditional Python tools:

```bash
# Create venv with standard Python
python3 -m venv .venv
source .venv/bin/activate

# Install YOUR dependencies first
pip install --upgrade pip
pip install -e .

# Then install vLLM (will respect your package versions)
bash scripts/install-vllm.sh
```

**Note:** The order matters! Install your dependencies before vLLM to avoid package conflicts.

### What You'll See

```
Model = meta-llama/Llama-3.2-1B, EP=False, TP=1, PP=1, DP=1, MaxTokens=256, Logprobs=1024, Prompts=1

[main] Launching CPU worker…
[cpu] VLLM_PLUGINS = cpu

[main] Launching RBLN worker…
[rbln] VLLM_PLUGINS = <unset>

────────────────────────────────────────────────────────────────────────────────
Prompt[0]: Hello, my name is
────────────────────────────────────────────────────────────────────────────────
 Generated text (CPU): len=256
Generated text (RBLN): len=256
Outliers (absΔ ≥ EPS): 3  (EPS=0.5)
        Vocab size: 128256
    Finite overlap: 1024
          max|Δ|: 0.000234
         mean|Δ|: 0.000012
          pearson: 0.999987  ✓

Logits-like (logprob) snippet — head…tail
  rbln  : [-8.1234, -7.9876, ..., -12.5678, -11.2345]
  golden: [-8.1256, -7.9888, ..., -12.5690, -11.2357]

Top-5 (logprob) argmax — RBLN vs GOLD
--------------------------------------------------------
Rank     R.idx      R.val        G.idx      G.val
   1     12345    -1.2345        12345    -1.2356
   2     23456    -2.3456        23456    -2.3467
   3     34567    -3.4567        34567    -3.4578
   4     45678    -4.5678        45678    -4.5689
   5     56789    -5.6789        56789    -5.6790
```

High Pearson correlation (>0.999) indicates excellent parity between CPU and RBLN implementations!

### Common Issues During Setup

**Issue:** `command not found: vllm-rbln-exec` after installation
```bash
# Solution: Activate your virtual environment
source .venv/bin/activate
```

**Issue:** Packages keep reinstalling (e.g., transformers, torch)
```bash
# Solution: Wrong installation order
# Your dependencies should be installed BEFORE vLLM
# Use: make install (which does the correct order)
# Or manually:
make clean-all
make venv
source .venv/bin/activate
make sync            # Install your deps first
make install-vllm    # Then install vLLM
```

**Issue:** `ValueError: 'aimv2' is already used by a Transformers config`
```bash
# Solution: Clean install with correct transformers version
rm -rf .venv vllm_source
uv venv
source .venv/bin/activate
make install  # Installs transformers <4.54 first, then vLLM
```

**Issue:** `No module named pip` when running `make install-vllm`
```bash
# Solution: The script auto-detects uv and uses it
# Just re-run: make install
```

**Issue:** Build fails with gcc errors
```bash
# Solution: Ensure gcc-12 or newer
gcc --version  # Should show 12.x or higher
sudo apt-get install -y gcc-12 g++-12
```

For more troubleshooting, see [Troubleshooting](#troubleshooting).

### Next Steps

- 📖 Read [Supported Models](#supported-models) to see all available models
- 🔧 Explore [Advanced Examples](#advanced-examples) for MoE models and parallelism
- 💾 Learn about [Caching](#caching) to speed up iterative testing
- 📊 Try [Profiling](#profiling) to analyze performance
- 🎨 Customize [Visualization](#visualization) options

---

## Overview

`vllm-rbln-exec` runs language model inference on both CPU and RBLN devices in separate processes with clean environments, then compares the outputs to verify parity. It provides detailed metrics including:

- Generated text comparison
- Logits/logprobs divergence analysis (max|Δ|, mean|Δ|, Pearson correlation)
- Top-K token comparison
- Outlier detection

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Supported Models](#supported-models)
  - [Advanced Examples](#advanced-examples)
- [Command-Line Options](#command-line-options)
- [Output Format](#output-format)
- [Caching](#caching)
- [Profiling](#profiling)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Features

- ✅ **Dual-device testing** - Runs vLLM on both CPU and RBLN in isolated processes
- ✅ **Comprehensive metrics** - Pearson correlation, L1 distances, outlier counts
- ✅ **Flexible configuration** - Supports various parallelism modes (TP, PP, DP, EP)
- ✅ **Result caching** - Cache CPU results for faster iteration
- ✅ **Multiple models** - Pre-configured support for Llama, Qwen, DeepSeek, and MoE models
- ✅ **Rich visualization** - Color-coded output with detailed logits inspection

## Requirements

### System Dependencies (Ubuntu/Debian)

```bash
sudo apt-get update -y
sudo apt-get install -y \
    gcc-12 g++-12 \
    cmake ninja-build git \
    libopenblas-dev \
    libtcmalloc-minimal4
```

### Python Requirements

- Python >= 3.9
- vLLM 0.9.1 (CPU build) - installed separately
- See `pyproject.toml` for other dependencies

## Installation

### Step 1: Install vLLM 0.9.1 (CPU)

This package requires vLLM 0.9.1 built for CPU with editable install. The installation process:
1. Creates a virtual environment
2. Installs your package dependencies first (including pinned `transformers` version)
3. Clones vLLM v0.9.1 source to `./vllm_source`
4. Installs vLLM build dependencies
5. Builds and installs vLLM in editable mode with `VLLM_TARGET_DEVICE=cpu`

**Important:** Dependencies are installed in a specific order to avoid version conflicts.

```bash
# Option A: One-command installation (recommended)
make install         # Creates venv, installs dependencies, then vLLM

# Option B: Step-by-step
make venv            # Create virtual environment
source .venv/bin/activate
make sync            # Install package dependencies FIRST
make install-vllm    # Then install vLLM (10-15 minutes, respects existing packages)
```

### Manual Installation

If `make` is not available:

```bash
# 1. Create and activate virtual environment
uv venv
source .venv/bin/activate

# 2. Install package dependencies FIRST (locks transformers version)
uv sync

# 3. Install vLLM (respects your pinned dependencies)
bash scripts/install-vllm.sh

# 4. Verify installation
python -c "import vllm; print(f'vLLM {vllm.__version__}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"
```

### Using pip instead of uv

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install package dependencies first
pip install -e .

# 3. Install vLLM
bash scripts/install-vllm.sh
```

**Why this order?** Installing your dependencies first ensures that vLLM respects your pinned package versions (especially `transformers`), avoiding reinstalls and version conflicts.

## Usage

### Basic Usage

```bash
# Run with default Llama 3.2 1B model
vllm-rbln-exec --model llama3.2-1b

# Run with custom prompts
vllm-rbln-exec --model llama3-8b --prompts "Once upon a time" "In the beginning"

# Generate more tokens
vllm-rbln-exec --model qwen3-1.7b --max-tokens 512
```

### Supported Models

| Model Name | Model ID | Expert Parallel |
|------------|----------|-----------------|
| `llama3.2-1b` | meta-llama/Llama-3.2-1B | No |
| `llama3-8b` | meta-llama/Meta-Llama-3-8B | No |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | No |
| `qwen1.5-moe-15b` | Qwen/Qwen1.5-MoE-A2.7B | Yes (--ep) |
| `qwen3-moe-30b` | Qwen/Qwen3-30B-A3B | Yes (--ep) |
| `qwen3-moe-235b` | Qwen/Qwen3-235B-A22B | Yes (--ep) |
| `deepseek-v2` | deepseek-ai/DeepSeek-V2-Lite | Yes (--ep) |
| `llama4-maverick` | meta-llama/Llama-4-Maverick-17B-128E | Yes (--ep) |

### Advanced Examples

```bash
# MoE model with expert parallel
vllm-rbln-exec --model qwen3-moe-30b --ep --tp 2 --pp 1

# Full vocabulary logprobs inspection
vllm-rbln-exec --model llama3-8b --logprobs -1 --inspect-logits

# Batch processing with caching
vllm-rbln-exec --model llama3.2-1b --batch 4 --use-cache --num-prompts 100

# Override model layers (for testing)
vllm-rbln-exec --model llama3-8b --num-hidden-layers 4

# Disable logits inspection
vllm-rbln-exec --model qwen3-1.7b --no-inspect-logits

# Profile execution
vllm-rbln-exec --model llama3-8b --profile
```

## Command-Line Options

### Model Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `llama3.2-1b` | Model name (see supported models) |
| `--max-model-len` | `40960` | Maximum model context length |
| `--num-hidden-layers` | None | Override number of hidden layers |
| `--trust-remote-code` | False | Trust remote code for custom models |

### Parallelism

| Option | Default | Description |
|--------|---------|-------------|
| `--tp` | `1` | Tensor parallel size (RBLN only) |
| `--pp` | `1` | Pipeline parallel size (RBLN only) |
| `--dp` | `1` | Data parallel size (RBLN only) |
| `--ep` | False | Enable expert parallel (required for MoE) |

### Generation

| Option | Default | Description |
|--------|---------|-------------|
| `--batch` | `1` | Batch size |
| `--max-tokens` | `256` | Tokens to generate per prompt |
| `--max-batched` | `128` | Max batched tokens |
| `--prompts` | None | Custom prompt list |
| `--num-prompts` | None | Number of prompts to use |

### Logprobs & Analysis

| Option | Default | Description |
|--------|---------|-------------|
| `--logprobs` | `1024` | 0=off, N=top-N, -1=full vocab |
| `--max-logprobs-cap` | `128256` | Engine-wide logprobs cap |
| `--inspect-logits` | True | Enable logits inspection |
| `--topk` | `5` | Top-K for argmax summary |

### Block Size

| Option | Default | Description |
|--------|---------|-------------|
| `--block-size-cpu` | `128` | KV cache block size for CPU |
| `--block-size-rbln` | `8192` | KV cache block size for RBLN |

### Visualization

| Option | Default | Description |
|--------|---------|-------------|
| `--no-color` | False | Disable ANSI colors |
| `--no-snippet` | False | Hide head/tail logits snippets |
| `--snippet-elems` | `6` | Elements in snippet |

### Caching & Profiling

| Option | Default | Description |
|--------|---------|-------------|
| `--use-cache` | False | Use cached CPU results |
| `--profile` | False | Enable torch profiler |

## Output Format

The tool provides detailed comparison output for each prompt:

```
────────────────────────────────────────────────────────────────────────────────
Prompt[0]: Hello, my name is
────────────────────────────────────────────────────────────────────────────────
 Generated text (CPU): len=256
Generated text (RBLN): len=256
Outliers (absΔ ≥ EPS): 12  (EPS=0.5)
        Vocab size: 128256
    Finite overlap: 1024
          max|Δ|: 0.000234
         mean|Δ|: 0.000012
          pearson: 0.999987

Logits-like (logprob) snippet — head…tail
  rbln  : [-8.1234, -7.9876, ..., -12.5678, -11.2345]
  golden: [-8.1256, -7.9888, ..., -12.5690, -11.2357]

Top-5 (logprob) argmax — RBLN vs GOLD
--------------------------------------------------------
Rank     R.idx      R.val        G.idx      G.val
   1     12345    -1.2345        12345    -1.2356
   2     23456    -2.3456        23456    -2.3467
   ...
```

## Project Structure

```
vllm-rbln-exec/
├── vllm_source/              # vLLM v0.9.1 editable install (gitignored)
├── src/
│   └── vllm_rbln_exec/
│       ├── __init__.py
│       ├── __main__.py       # Entry point
│       ├── parity_runner.py  # Main comparison logic
│       └── setup.py          # vLLM installation helper
├── scripts/
│   └── install-vllm.sh       # vLLM installation script
├── cache/                    # CPU result cache (gitignored)
├── profile/                  # Profiler outputs (gitignored)
├── pyproject.toml
├── Makefile
└── README.md
```

## Caching

CPU results can be cached to speed up iteration during RBLN development:

```bash
# First run - generates and caches CPU results
vllm-rbln-exec --model llama3-8b --use-cache

# Subsequent runs - reuses cached CPU results
vllm-rbln-exec --model llama3-8b --use-cache
```

Cache keys include:
- Model name
- Number of hidden layers (if overridden)
- Max tokens
- Logprobs setting
- Number of prompts
- Prompt content hash

## Profiling

Enable torch profiler to analyze performance:

```bash
vllm-rbln-exec --model llama3-8b --profile
```

Profiles are saved to `./profile/cpu_*` and `./profile/rbln_*`.

## Environment Variables

The tool automatically sets these based on device:

**CPU:**
- `VLLM_PLUGINS=cpu`
- `VLLM_USE_V1=0`

**RBLN:**
- `RBLN_KERNEL_MODE=triton`
- `USE_VLLM_MODEL=1`
- `VLLM_USE_V1=0`
- `VLLM_DISABLE_COMPILE_CACHE=1`

## Troubleshooting

### vLLM import errors

```bash
# Verify installation
python -c "import vllm; print(vllm.__version__)"

# Reinstall if needed
rm -rf vllm_source .venv
make install
```

### Build failures

- Ensure gcc-12 or newer: `gcc --version`
- Increase system RAM (build needs ~8GB)
- Reduce parallel jobs: `export MAX_JOBS=2`

### Transformers version conflicts

If you see `'aimv2' is already used by a Transformers config`:

```bash
# Remove transformers from pyproject.toml dependencies
# Let vLLM manage its own transformers version
rm -rf .venv
uv sync
```

**Or** keep your pinned version by ensuring correct installation order:
```bash
# This package pins transformers>=4.43,<4.54.0 in pyproject.toml
# The installation order ensures vLLM respects this:
# 1. make sync    (installs your pinned transformers)
# 2. make install-vllm (skips transformers, uses yours)
```

The `install-vllm.sh` script detects if transformers is already installed and skips it, preventing version conflicts.

### Cache issues

```bash
# Clear cache
rm -rf cache/

# Disable cache
vllm-rbln-exec --model llama3-8b  # without --use-cache
```

## Development

```bash
# Create virtual environment
make venv

# Install with dev dependencies
uv sync --extra dev

# Run tests
make test

# Run linters
make lint

# Clean build artifacts
make clean

# Complete cleanup (including venv and vLLM source)
make clean-all

# See all available commands
make help
```

## Contributing

This is a tool for internal vLLM RBLN plugin development and testing.

## License

Apache 2.0

## Related Documentation

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html)
- [Rebellions NPU](https://rebellions.ai/)

## Notes

- CPU and RBLN workers run in **separate processes** with isolated environments
- vLLM is installed in **editable mode** at `./vllm_source` for plugin development
- The tool uses `spawn` multiprocessing context for clean process isolation
- Logprobs are treated as "logits-like" vectors for comparison metrics
