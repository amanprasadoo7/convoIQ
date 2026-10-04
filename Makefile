.DEFAULT_GOAL := help
.PHONY: help setup setup-ml setup-guard setup-all check-ollama free-gpu baseline api test lint clean

# source .venv/bin/activate
# deactivate

help:  ## Show all commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

setup:  ## Base install (Day 0)
	uv sync

setup-ml:  ## Add torch + transformers (Day 2)
	uv sync --group ml

setup-guard:  ## Add Presidio PII tools (Day 4)
	uv sync --group guard

setup-all:  ## Everything, all groups
	uv sync --all-groups

check-ollama:  ## Verify Ollama is up and the models we need exist
	@ollama list | grep -q "llama3.2:3b" || (echo "missing llama3.2:3b" && exit 1)
	@ollama list | grep -q "llama-guard3:1b" || (echo "missing llama-guard3:1b" && exit 1)
	@ollama list | grep -q "nomic-embed-text" || (echo "missing nomic-embed-text" && exit 1)
	@echo "Ollama OK"

free-gpu:  ## Unload Ollama models so training gets the full 6GB VRAM
	-ollama stop llama3.2:3b
	-ollama stop llama-guard3:1b
	-ollama stop gemma4:e4b
	@nvidia-smi --query-gpu=memory.used,memory.total --format=csv

baseline:  ## Day 1: prompted-LLM intent baseline
	uv run python -m convintel.baseline

api:  ## Run the FastAPI server with auto-reload
	uv run uvicorn convintel.api:app --reload --port 8000

test:  ## Run tests
	uv run pytest -q

lint:  ## Lint and format check
	uv run ruff check .
	uv run ruff format --check .

clean:  ## Remove caches (keeps .venv)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +