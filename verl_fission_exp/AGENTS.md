# Repository Guidelines

## Project Structure & Module Organization
- Source code: `verl/` (core library). Configs: `verl/trainer/config/*.yaml`. Version: `verl/version/version`.
- Tests: `tests/` with category subfolders; CPU-only tests end with `*_on_cpu.py`.
- Docs: `docs/` (Sphinx). Examples and utilities: `examples/`, `scripts/`. Recipes: `recipe/`. Container assets: `docker/`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install for dev: `pip install -e .[test]`
- Lint/format/type check: `pre-commit install && pre-commit run -a` (runs ruff, ruff-format, mypy, config/doc/license checks).
  - Direct calls: `ruff check . && ruff format` and `mypy verl`
- Run tests:
  - All: `pytest -s --asyncio-mode=auto tests/`
  - CPU subset: `pytest -s --asyncio-mode=auto tests -k 'on_cpu'`
- Build docs: `pip install -r docs/requirements-docs.txt && make -C docs html`
- Generate trainer config after YAML changes: `scripts/generate_trainer_config.sh`

## Coding Style & Naming Conventions
- Python ≥ 3.10, 4-space indent, max line length 120 (see `pyproject.toml`).
- Use type hints; add docstrings for public APIs (checked in pre-commit).
- Naming: modules/functions `snake_case`; classes `CamelCase`; constants `UPPER_SNAKE_CASE`.
- Imports sorted via ruff/isort; avoid unused imports and broad wildcards.
- New Python files must include the Apache-2.0 header used in this repo.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`. Place new tests under matching subpackages in `tests/`.
- File naming: `test_*.py`; CPU-only: `*_on_cpu.py`. Keep tests deterministic and fast; put heavy GPU/e2e tests under `tests/special_*`.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`). Write clear, scoped messages.
- PRs include: description, motivation, linked issues, test plan/output, and docs updates if user-facing.
- Before pushing: `pre-commit run -a` passes and `pytest` green locally. For reproducible runs, consider the images under `.github/workflows` or `docker/`.

## 重要注意事项
- 后续回答我的问题都需要使用中文进行回答，除非是一些专业的术语或者框架中相关的参数可以使用英语
- 因为我是个初学者，对于相关的内容和概念不是很了解，所以在回答我的问题的时候务必详细一些，不要节省你的token，像是教授学生一样详细的回答的问题，并给我介绍相关的知识点，让我可以更加详细完整的了解这个框架。