# AGENTS.md

## Current state

Day 1 is in progress. What exists: the scaffold, `loaders/banking77.py` (Banking77 loader), and
`src/baseline.py` (prompted-LLM intent baseline, smoke-tested only — not yet run at scale).
Everything else in `docs/mine/updated.md` is *planned, not built*. Do not assume a module, model, or
endpoint referenced there exists; check first.

## Commands

All workflow goes through `make` (`.DEFAULT_GOAL := help`).

| Command | Runs |
| --- | --- |
| `make setup` | `uv sync` — dev group only (`default-groups = ["dev"]`) |
| `make setup-ml` | adds `ml` group (torch, transformers, accelerate) |
| `make setup-guard` | adds `guard` group (Presidio, spacy) |
| `make setup-all` | all groups |
| `make lint` | `ruff check .` then `ruff format --check .` |
| `make test` | `uv run pytest -q` |
| `make api` | `uvicorn src.api:app --reload --port 8000` |

**No make target exists for the `analytics`, `db`, or `obs` groups.** Install them directly:
`uv sync --group analytics`.

### Non-obvious gotchas

- **`uv run` silently re-locks `uv.lock`.** The committed lockfile is stale relative to
  `pyproject.toml` (it predates the `ml`/`guard`/`analytics`/`db`/`obs` groups), so the first
  `uv run` or `uv sync` rewrites it — roughly a 4600-line diff. Expected, not a bug, but don't let it
  ride along in an unrelated commit.
- **`make lint` formats Python code blocks inside Markdown**, including `README.md` and
  `docs/mine/*.md`. A badly formatted ```python snippet in a doc breaks lint. Fix with
  `uv run ruff format .`
- **No `select` is configured, so ruff's defaults apply — and they are broad** (~413 rules in ruff
  0.16). Expect failures from rules that aren't obviously on: isort (`I001`), blind-except
  (`BLE001`), bugbear, `TRY`, `SIM`, `UP`, `PLR` and more. Notably **`T201` (print) is *not*
  enabled**, so `print()` is fine. `loaders/banking77.py` carries a deliberate `# noqa: BLE001` on its
  fallback — keep that, the blind catch is intentional.
- ruff also lint-checks `pyproject.toml` itself. Line length is 100.
- **`make test` exits 5 when zero tests are collected**, which make reports as a failure. Expected
  until tests exist.
- **No typechecker** (no mypy/pyright) and **no CI or pre-commit hooks** are configured. Don't
  introduce one uninvited.

### Single test

```
uv run pytest path/to/test_x.py::test_name
```

`asyncio_mode = "auto"`, so async tests need **no** `@pytest.mark.asyncio` decorator.

## Package layout and two naming traps

- The importable package is **`src/` at the repo root** — import name `src`, modules `src.baseline`,
  `src.inspect`, `src.config`, `src.explore_dataset`.
- `pyproject.toml` sets `[tool.uv] package = false`, so the project is never pip-installed.
  `python -m src.baseline` resolves only because `uv run` places the repo root (cwd) on `sys.path`.
  **`src/` must stay at the repo root** — nesting it deeper (e.g. `pkg/src/`) will not import without
  extra config.
- **Never name a repo-root directory `datasets/`.** The installed HuggingFace `datasets` package is a
  *regular* package, and a regular package always beats a local namespace directory of the same
  name. So `from datasets.banking77 import ...` fails with `ModuleNotFoundError` — Python resolves
  `datasets` to site-packages and never looks at the local folder. Adding an `__init__.py` makes it
  worse: then the local dir shadows site-packages and the loader's own
  `from datasets import load_dataset` breaks. That is why the loader lives in **`loaders/`**.
- Historical note: the package was `convoiq/`, and before that the Makefile referenced
  `convintel.*`, which never existed. `make api` still fails because there is no `src/api.py` yet.

## Local environment

- GPU is an **RTX 3050 with 6GB VRAM**. Run `make free-gpu` before any training or heavy inference, or
  Ollama keeps the models resident and you OOM.
- Ollama (which exposes an OpenAI-compatible API) is the local LLM provider. `make check-ollama`
  requires `llama3.2:3b`, `llama-guard3:1b`, and `nomic-embed-text`.
- `gemma4:e4b` is pulled locally but is 9.6GB and cannot fit in 6GB VRAM — don't plan around it.
- Dependency groups are staged **by day on purpose** (see the plan's Timeline). Sync only the group
  the current step needs rather than running `setup-all` up front.

## Claims-integrity constraints

From the plan's "Problems Identified and Corrections". These are deliberate decisions, not gaps.
Overstating any of them in code, docs, or commit messages is the main way to damage this project.

- **Banking77 contains single utterances, not multi-turn conversations.** Use it for intent
  classification only. Use Twitter Customer Support (`twcs`) for conversation threads and sentiment.
  Never present Banking77 as a conversation dataset.
- **The bot-vs-human A/B test is simulated** — no public dataset has controlled bot/human outcomes.
  Label it simulated and state the assumptions.
- **Do not claim Azure OpenAI or Vertex AI usage.** Claim only that a provider abstraction exists that
  could swap providers. The same applies to KV cache, HPA/autoscaling, and LangGraph state — all out
  of scope here.
- **Evaluation is never cut** when time runs short. Cut order is drift detection → Grafana → LightGBM
  → Whisper. Report both micro and macro F1, since Banking77's 77 classes are imbalanced.

## Reference docs

- **`docs/mine/updated.md`** — the authoritative design spec and 12-day timeline (~1800 lines).
  Everything else derives from it; read the relevant section before implementing a stage. It is
  **gitignored** (local working notes) — don't commit, un-ignore, or relocate it unasked.
- `docs/mine/building_thoughts.md` — earlier, rougher draft. Superseded by `updated.md`; prefer
  `updated.md` on any conflict.
- `docs/arch.png`, `docs/timeline.png` — figures the plan references.
- `docs/links.txt` is gitignored (a personal share link). Leave it alone.
- `README.md` is **empty (0 bytes)** despite being the declared `readme` in `pyproject.toml`. It's a
  required final deliverable.