.PHONY: help install install-vllm sync clean test lint venv patch-models

help:
	@echo "vllm-rbln-exec setup commands:"
	@echo ""
	@echo "  make install       - Full installation (creates venv, installs vLLM + dependencies)"
	@echo "  make venv          - Create virtual environment only"
	@echo "  make install-vllm  - Install only vLLM 0.9.1 (CPU)"
	@echo "  make sync          - Sync dependencies with uv"
	@echo "  make patch-models  - Patch vLLM models for num_hidden_layers override"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make clean         - Clean build artifacts"

venv:
	@if [ ! -d .venv ]; then \
		echo "Creating virtual environment with uv..."; \
		uv venv; \
		echo "✓ Virtual environment created at .venv"; \
		echo ""; \
		echo "Activate it with:"; \
		echo "  source .venv/bin/activate"; \
	else \
		echo "✓ Virtual environment already exists at .venv"; \
	fi

install: venv sync install-vllm patch-models
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  ✅ Installation complete!                                 ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate your environment:"
	@echo "     source .venv/bin/activate"
	@echo ""
	@echo "  2. Run your first test:"
	@echo "     vllm-rbln-exec --model llama3.2-1b"

install-vllm: venv
	@if [ ! -f .venv/bin/activate ]; then \
		echo "❌ Virtual environment not found. Run 'make venv' first."; \
		exit 1; \
	fi
	@echo "Installing vLLM (this will take 10-15 minutes)..."
	@bash scripts/install-vllm.sh

patch-models:
	@echo "Patching vLLM models for num_hidden_layers override..."
	@python3 scripts/patch_vllm_models.py

sync: venv
	@if [ ! -f .venv/bin/activate ]; then \
		echo "❌ Virtual environment not found. Run 'make venv' first."; \
		exit 1; \
	fi
	@uv sync

test:
	@uv run pytest tests/

lint:
	@uv run ruff check src/
	@uv run mypy src/

clean:
	@rm -rf build/ dist/ *.egg-info .pytest_cache .coverage .mypy_cache .ruff_cache
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf src/*.egg-info
	@echo "✓ Build artifacts cleaned"

clean-all: clean
	@echo "Removing virtual environment and vLLM source..."
	@rm -rf .venv vllm_source cache/ profile/
	@echo "✓ Complete cleanup done"