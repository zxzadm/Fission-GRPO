# Repository Guidelines

## Project Structure & Module Organization

- `bfcl_eval/`: Python package (CLI + evaluation pipeline).
  - `bfcl_eval/__main__.py`: `bfcl` CLI entrypoint.
  - `bfcl_eval/model_handler/`: model integrations (API + local backends).
  - `bfcl_eval/eval_checker/`: evaluation runners and checkers.
  - `bfcl_eval/data/`: datasets and test definitions.
  - `bfcl_eval/scripts/`: helper scripts.
- Top-level docs: `README.md`, `CONTRIBUTING.md`, `TEST_CATEGORIES.md`, `LOG_GUIDE.md`.
- Runtime outputs (created under the project root): `result/` (model outputs) and `score/` (metrics + CSVs).

## Build, Test, and Development Commands

Python >= 3.10 is required (`pyproject.toml`).

- Install for development: `pip install -e .`
- Optional extras:
  - Local OSS backends: `pip install -e '.[oss_eval_sglang]'` or `pip install -e '.[oss_eval_vllm]'`
  - WandB logging: `pip install -e '.[wandb]'`
- List supported items:
  - `bfcl models`
  - `bfcl test-categories`
- Run an eval (end-to-end):
  - Generate: `bfcl generate --model MODEL --test-category CATEGORY`
  - Evaluate: `bfcl evaluate --model MODEL --test-category CATEGORY`
  - Show table: `bfcl scores`

## Coding Style & Naming Conventions

- Follow existing patterns in `bfcl_eval/`: 4-space indentation, `snake_case` for functions/files, `PascalCase` for classes.
- Prefer type hints for new APIs and keep interfaces consistent with `bfcl_eval/model_handler/base_handler.py`.
- Keep changes scoped: dataset edits in `bfcl_eval/data/`, model logic in `bfcl_eval/model_handler/`, scoring logic in `bfcl_eval/eval_checker/`.

## Testing Guidelines

There is no standalone unit test suite in this checkout; validate changes by running a small evaluation.

- Use targeted runs during development:
  - Copy `bfcl_eval/test_case_ids_to_generate.json.example` to `./test_case_ids_to_generate.json`
  - Run `bfcl generate --run-ids ...` to regenerate only listed IDs, then `bfcl evaluate ...`

## Commit & Pull Request Guidelines

- This directory does not include `.git`, so commit conventions can’t be inferred here. Prefer Conventional Commits (e.g., `fix: ...`, `feat(model): ...`) or match the upstream repo’s style.
- PRs should include: what changed, how to reproduce (`bfcl generate/evaluate` commands), and any new/updated model config entries.
- Do not commit secrets or local artifacts: keep `.env`, `result/`, and `score/` out of version control.

## Configuration & Security Notes

- Copy `bfcl_eval/.env.example` to `.env` and set required API keys.
- If installed from PyPI, set `BFCL_PROJECT_ROOT` so outputs and `.env` resolve correctly (see `README.md`).
