# Repository Guidelines

## Project Structure & Module Organization

`auto_tune_vllm/` is the package: `core/` holds study and trial models,
`execution/` runs local or Ray trials, `benchmarks/` defines providers, and
`cli/` exposes the Typer interface. The optional tuner is in `agent/`; packaged
vLLM defaults are in `schemas/`. Put YAML and runnable samples in `examples/`,
user-facing documentation in `docs/`, and smoke coverage in `test_installation.py`.

## Build, Test, and Development Commands

Use Python 3.10 or newer. Install the editable package and development tools:

```bash
pip install -e ".[dev]"
pip install -e ".[agent]"       # when changing agentic tuning features
pytest -q                        # run the test suite
python test_installation.py       # run import and basic API smoke checks
ruff check . && ruff format --check .
basedpyright
auto-tune-vllm --help
```

Run `ruff format .` to apply formatting. Keep tests independent of GPU, Ray,
PostgreSQL, and vLLM infrastructure where possible.

## Coding Style & Naming Conventions

Use four-space indentation, double quotes, and an 88-character line limit.
Ruff enforces error (`E`), Pyflakes (`F`), and import-sort (`I`) rules. Use
`snake_case` for modules, functions, variables, YAML keys, and CLI options;
use `PascalCase` for classes; and name tests like `test_config_rejects_invalid_model`.
Type new public and non-trivial internal APIs; basedpyright excludes `agent/`.

## Testing Guidelines

Write pytest tests named `test_*.py` with test functions beginning `test_`.
Add focused tests near their code when a test package is introduced, or extend
root smoke coverage for installation/public-API changes. Mock backend boundaries
rather than launching real clusters. Run relevant pytest tests, lint, and type
checks before submitting.

## Commit & Pull Request Guidelines

Recent history follows short, imperative Conventional Commit-style subjects,
for example `feat: add constraint sampling`, `fix: baseline bugs`, or
`refactor: remove schema usage`. Keep commits narrowly scoped. Pull requests
should explain the behavior change, validation performed, and related issue;
include config snippets or terminal output for CLI/documentation changes.

## Configuration & Security

Do not commit credentials, kubeconfig files, PostgreSQL connection strings, or
cluster endpoints. Use sanitized YAML examples and document required settings.
